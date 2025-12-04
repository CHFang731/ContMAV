import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import sys

# 確保可以導入 src 下的模組
sys.path.append(os.getcwd())

from src.args import ArgumentParser
from src.build_model import build_model
from src.datasets.cityscapes.cityscapes import CityscapesBase

def colorize_mask(mask):
    """將分割結果 (0-18) 轉換為 Cityscapes 的彩色圖片"""
    palette = CityscapesBase.CLASS_COLORS_REDUCED
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(palette):
        color_mask[mask == label_id] = color
    return color_mask

def save_visualization(orig_img_pil, pred_mask_np, save_path):
    """
    只儲存疊加圖 (Overlay)
    """
    # 1. 準備預測結果圖 (彩色)
    colored_mask = colorize_mask(pred_mask_np)
    mask_img_pil = Image.fromarray(colored_mask).convert("RGB")

    # 2. 準備疊加圖 (Overlay)
    # alpha=0.5 代表 50% 原圖 + 50% 預測結果
    overlay_img_pil = Image.blend(orig_img_pil, mask_img_pil, alpha=0.5)

    # 3. 只儲存疊加圖
    overlay_img_pil.save(save_path)
    print(f"Success! Overlay image saved to {save_path}")

def main():
    parser = ArgumentParser(description="Inference for ContMAV")
    parser.set_common_args()
    
    parser.add_argument("--img_path", type=str, required=True, help="輸入圖片的路徑")
    parser.add_argument("--ckpt_path", type=str, required=True, help="訓練好的模型權重路徑")
    parser.add_argument("--save_path", type=str, default="vis_overlay.png", help="結果儲存路徑")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 建立模型
    n_classes = 19
    print(f"Building model (Encoder: {args.encoder})...")
    model, _ = build_model(args, n_classes=n_classes)
    model.to(device)
    model.eval()

    # 載入權重
    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"Warning: Strict loading failed, trying with strict=False.")
        model.load_state_dict(new_state_dict, strict=False)

    # 處理圖片
    print(f"Processing image {args.img_path}...")
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    # 讀取並保存原始 PIL 物件供後續視覺化使用
    orig_image_pil = Image.open(args.img_path).convert('RGB')
    orig_w, orig_h = orig_image_pil.size
    
    preprocess = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(orig_image_pil).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        # 取得預測結果
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 將預測結果 Resize 回原始大小
    pred_mask_orig_size = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # 呼叫新的視覺化儲存函數
    save_visualization(orig_image_pil, pred_mask_orig_size, args.save_path)

if __name__ == "__main__":
    main()