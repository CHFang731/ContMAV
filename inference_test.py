import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

# 確保可以導入 src 下的模組
sys.path.append(os.getcwd())

from src.args import ArgumentParser
from src.build_model import build_model
from src.datasets.cityscapes.cityscapes import CityscapesBase


# ---------------------------------------------------------
# 工具函式：上色、疊圖
# ---------------------------------------------------------

def colorize_mask(mask):
    """將分割結果 (0-18) 轉換為 Cityscapes 的彩色圖片"""
    palette = CityscapesBase.CLASS_COLORS_REDUCED
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(palette):
        color_mask[mask == label_id] = color
    return color_mask


def overlay_segmentation(orig_img_pil, seg_mask_np, alpha=0.5):
    """產生 segmentation 疊加圖"""
    colored_mask = colorize_mask(seg_mask_np)
    mask_img_pil = Image.fromarray(colored_mask).convert("RGB")
    overlay_img_pil = Image.blend(orig_img_pil, mask_img_pil, alpha=alpha)
    return overlay_img_pil


def overlay_unknown(orig_img_pil, unk_score_np, tau=0.5):
    """
    在原圖上疊一層紅色透明遮罩表示 unknown。
    unk_score_np: [H, W]，數值越大 = 越像 unknown
    """
    orig = np.array(orig_img_pil).astype(np.float32) / 255.0
    h, w, _ = orig.shape

    # normalize 到 0~1
    s = unk_score_np.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-6)

    # binary mask (for 強烈 unknown 區域)
    bin_mask = (s > tau).astype(np.float32)

    # 紅色遮罩
    red = np.zeros_like(orig)
    red[..., 0] = 1.0  # R channel = 1

    # heatmap 強度
    alpha_map = (0.6 * s)[..., None]  # 0~0.6 之間

    overlay = orig * (1 - alpha_map) + red * alpha_map
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay), bin_mask


# ---------------------------------------------------------
# MAV / STD 載入（盡量支援多種格式）
# ---------------------------------------------------------

def load_mavs_stds_from_ckpt(ckpt, num_classes, device):
    """
    從 checkpoint 中讀出 mavs / stds。
    支援：
      - tensor: [C, D]
      - list/tuple: 長度 C，每個 element 是 [D]
      - dict: key 可能是 int / str / 類別名稱
    如果格式不認得，退化成 (None, None)，後面會用 entropy/norm 當 unknown 分數。
    """
    if "mavs" not in ckpt or "stds" not in ckpt:
        print("[WARN] checkpoint 中沒有 'mavs' / 'stds'，將只用 entropy/norm 當 unknown 分數。")
        return None, None

    mavs_raw = ckpt["mavs"]
    stds_raw = ckpt["stds"]

    # case 1: 直接就是 tensor [C, D]
    if isinstance(mavs_raw, torch.Tensor) and isinstance(stds_raw, torch.Tensor):
        mavs = mavs_raw.to(device)
        stds = stds_raw.to(device)
        print(f"[INFO] mavs/stds 以 tensor 格式載入: {mavs.shape}")
        return mavs, stds

    # case 2: list/tuple of tensor
    if isinstance(mavs_raw, (list, tuple)) and isinstance(stds_raw, (list, tuple)):
        if len(mavs_raw) == num_classes and len(stds_raw) == num_classes:
            mavs = torch.stack(mavs_raw, dim=0).to(device)
            stds = torch.stack(stds_raw, dim=0).to(device)
            print(f"[INFO] mavs/stds 以 list/tuple 格式載入: {mavs.shape}")
            return mavs, stds
        else:
            print("[WARN] mavs/stds 是 list/tuple，但長度 != num_classes，跳過 Gaussian。")
            return None, None

    # case 3: dict：key 可能是 int / str / 類別名稱
    if isinstance(mavs_raw, dict) and isinstance(stds_raw, dict):
        keys_all = list(mavs_raw.keys())
        print(f"[INFO] mavs/stds 是 dict，keys = {keys_all}")

        # 如果都是 int，就用 0..C-1
        if all(isinstance(k, int) for k in keys_all):
            keys = list(range(num_classes))
        else:
            # 不是 int，可能是 str / 類別名稱
            # 先試 CityscapesBase.CLASS_NAMES_REDUCED（若有）
            if hasattr(CityscapesBase, "CLASS_NAMES_REDUCED"):
                keys = CityscapesBase.CLASS_NAMES_REDUCED
            else:
                # 保守作法：用字母排序，至少兩邊順序一致
                keys = sorted(mavs_raw.keys())

        mav_list = []
        std_list = []
        for k in keys:
            if k not in mavs_raw:
                print(f"[WARN] mavs/stds dict 中找不到 key={k}，跳過 Gaussian。")
                return None, None
            mv = mavs_raw[k]
            sd = stds_raw[k]
            if mv.dim() == 1:
                mv = mv.unsqueeze(0)
            if sd.dim() == 1:
                sd = sd.unsqueeze(0)
            mav_list.append(mv)
            std_list.append(sd)

        mavs = torch.cat(mav_list, dim=0).to(device)   # [C, D]
        stds = torch.cat(std_list, dim=0).to(device)
        print(f"[INFO] mavs/stds 以 dict 格式載入: {mavs.shape}")
        return mavs, stds

    print("[WARN] 無法辨識 mavs/stds 的格式，將只用 entropy/norm 當 unknown 分數。")
    return None, None


# ---------------------------------------------------------
# 主程式
# ---------------------------------------------------------

def main():
    parser = ArgumentParser(description="Open-World Inference for ContMAV")
    parser.set_common_args()

    parser.add_argument("--img_path", type=str, required=True, help="輸入圖片路徑")
    parser.add_argument("--ckpt_path", type=str, required=True, help="訓練好的模型權重路徑")
    parser.add_argument("--save_seg_overlay", type=str, default="seg_overlay.png",
                        help="Cityscapes segmentation 疊加圖輸出路徑")
    parser.add_argument("--save_unk_overlay", type=str, default="unk_overlay.png",
                        help="Unknown heatmap 疊加圖輸出路徑")
    parser.add_argument("--save_unk_mask", type=str, default="unk_mask.png",
                        help="Unknown binary mask (0/255) 輸出路徑")
    parser.add_argument("--xi", type=float, default=1.0,
                        help="contrastive norm 閾值 (unknown score 用)")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="unknown 分數二值化閾值 (0~1)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # 1. 建立模型
    # -------------------------
    n_classes = 19
    print(f"Building model (encoder: {args.encoder}, num_classes={n_classes})...")
    model, _ = build_model(args, n_classes=n_classes)
    model.to(device)
    model.eval()

    # -------------------------
    # 2. 讀取 checkpoint + 修正 key
    # -------------------------
    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 先把舊版名稱改成新版名稱（context_features -> context_module.features）
    fixed_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = new_k.replace("context_features.", "context_module.features.")
        new_k = new_k.replace("context_final_conv.", "context_module.final_conv.")
        new_k = new_k.replace("module.", "")  # 移除 DataParallel 前綴
        fixed_state_dict[new_k] = v

    try:
        model.load_state_dict(fixed_state_dict, strict=True)
        print("[INFO] state_dict strict=True 載入成功。")
    except Exception as e:
        print(f"[WARN] strict=True 載入失敗，改用 strict=False. error = {e}")
        model.load_state_dict(fixed_state_dict, strict=False)

    # 讀取 MAV / STD
    mavs, stds = load_mavs_stds_from_ckpt(ckpt, n_classes, device)

    # -------------------------
    # 3. 讀圖 + 前處理
    # -------------------------
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    orig_image_pil = Image.open(args.img_path).convert("RGB")
    orig_w, orig_h = orig_image_pil.size

    preprocess = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(orig_image_pil).unsqueeze(0).to(device)

    # -------------------------
    # 4. 推論：取得兩個 decoder 輸出
    # -------------------------
    with torch.no_grad():
        out = model(input_tensor)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            seg_logits, ow_feats = out        # seg_logits: [1,C,h,w], ow_feats: [1,C,h,w]
        else:
            seg_logits = out
            ow_feats = None
            print("[WARN] 模型只輸出一個 tensor，沒有 contrastive decoder。")

    # -------------------------
    # 5. closed-set segmentation 疊圖
    # -------------------------
    seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    seg_pred_orig = cv2.resize(seg_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    seg_overlay = overlay_segmentation(orig_image_pil, seg_pred_orig, alpha=0.5)
    seg_overlay.save(args.save_seg_overlay)
    print(f"[OK] Saved segmentation overlay to {args.save_seg_overlay}")

    # -------------------------
    # 6. 計算 unknown 分數
    # -------------------------
    # 6.1 semantic (Gaussian / entropy)
    seg_logits_up = torch.nn.functional.interpolate(
        seg_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )  # [1,C,H,W]
    seg_logits_up = seg_logits_up[0]  # [C,H,W]

    C, H, W = seg_logits_up.shape
    logits_flat = seg_logits_up.permute(1, 2, 0).reshape(-1, C)  # [N, C]

    sem_unknown = None
    if mavs is not None and stds is not None and mavs.shape[0] == C and mavs.shape[1] == C:
        # 使用 Gaussian 分數（假設 D = C = logits 維度）
        probs = torch.softmax(seg_logits_up.unsqueeze(0), dim=1)[0]  # [C,H,W]
        pred_cls = probs.argmax(dim=0).reshape(-1)                   # [N]

        mu = mavs[pred_cls]              # [N, C]
        sigma2 = (stds[pred_cls] ** 2) + 1e-6

        diff = logits_flat - mu
        dist2 = (diff * diff / sigma2).sum(dim=1)                    # [N]

        sim = torch.exp(-0.5 * dist2)    # 越接近 1 越像已知
        sem_unknown = 1.0 - sim          # 越接近 1 越像 unknown
        sem_unknown = sem_unknown.reshape(H, W)
        print("[INFO] 使用 Gaussian MAV 計算 semantic unknown 分數。")
    else:
        # fallback: 用 entropy 當 unknown 分數
        log_probs = torch.log_softmax(seg_logits_up, dim=0)          # [C,H,W]
        ent = -(torch.exp(log_probs) * log_probs).sum(dim=0)         # [H,W]
        ent_min, ent_max = ent.min(), ent.max()
        sem_unknown = (ent - ent_min) / (ent_max - ent_min + 1e-6)
        print("[WARN] 未使用 Gaussian MAV，改用 entropy 當 semantic unknown 分數。")

    # 6.2 contrastive decoder 的 unknown 分數（norm 反比）
    if ow_feats is not None:
        ow_up = torch.nn.functional.interpolate(
            ow_feats, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        ow_up = ow_up[0]                                        # [C,H,W]
        feats_flat = ow_up.permute(1, 2, 0).reshape(-1, C)      # [N,C]
        norms = torch.linalg.norm(feats_flat, dim=1)            # [N]

        xi = args.xi
        con_unknown = torch.clamp(1.0 - norms / xi, 0.0, 1.0)   # norm 越小越 unknown
        con_unknown = con_unknown.reshape(H, W)
        print("[INFO] 使用 contrastive feature norm 計算 unknown 分數。")
    else:
        con_unknown = sem_unknown.clone()
        print("[WARN] 沒有 contrastive decoder，unknown 分數只來自 semantic 部分。")

    # 6.3 融合兩種 unknown 分數
    unk_score = 0.5 * (sem_unknown + con_unknown)               # [H,W]
    unk_score_np = unk_score.cpu().numpy()

    # -------------------------
    # 7. 可視化 unknown（疊圖 + binary mask）
    # -------------------------
    unk_overlay_pil, bin_mask = overlay_unknown(
        orig_image_pil, unk_score_np, tau=args.tau
    )
    unk_overlay_pil.save(args.save_unk_overlay)
    print(f"[OK] Saved unknown overlay to {args.save_unk_overlay}")

    bin_mask_img = (bin_mask * 255).astype(np.uint8)
    Image.fromarray(bin_mask_img).save(args.save_unk_mask)
    print(f"[OK] Saved unknown binary mask to {args.save_unk_mask}")


if __name__ == "__main__":
    main()
