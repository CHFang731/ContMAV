########################################################
#       author: omitted for anonymous submission       #
#     credits and copyright coming upon publication    #
########################################################

import os
import glob
import numpy as np
import cv2
from ..dataset_base import DatasetBase
from .cityscapes import CityscapesBase

class Cityscapes(CityscapesBase, DatasetBase):
    def __init__(
        self,
        data_dir=None,
        n_classes=19,
        split="train",
        with_input_orig=False,
        overfit=False,
        classes=19,
    ):
        super(Cityscapes, self).__init__()
        assert split in self.SPLITS
        print(f"Initializing Cityscapes Dataset: Split={split}, Classes={n_classes}")

        self._n_classes = classes
        self._split = split
        self._with_input_orig = with_input_orig
        self._cameras = ["camera1"]
        self.overfit = overfit

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"
            self._data_dir = data_dir

            # === 修改 1: 處理 val 與 valid 的命名差異 ===
            # 您的資料夾名稱是 'valid'，但 split 參數傳入的是 'val'
            folder_split = "valid" if split == "val" else split

            # 輔助函數：讀取 txt
            def _loadtxt(fname):
                path = os.path.join(data_dir, fname)
                if not os.path.exists(path):
                    # 嘗試 fallback，如果 val_rgb.txt 不存在，試試 valid_rgb.txt
                    alt_path = path.replace("val_", "valid_")
                    if os.path.exists(alt_path):
                        print(f"Notice: Using {alt_path} instead of {path}")
                        path = alt_path
                    else:
                        raise FileNotFoundError(f"Could not find list file: {path}")
                with open(path, 'r') as f:
                    return [line.strip() for line in f.readlines()]

            # === 修改 2: 讀取檔案列表 ===
            rgb_list_file = f"{self._split}_rgb.txt"
            print(f"Loading image list from: {rgb_list_file}")
            rgb_rel_paths = _loadtxt(rgb_list_file)

            label_list_file = f"{self._split}_labels_{self._n_classes}.txt"
            print(f"Loading label list from: {label_list_file}")
            label_rel_paths = _loadtxt(label_list_file)

            # === 修改 3: 適配您的資料夾結構 (rgb 和 labels_19) ===
            # 您的結構是: root/valid/rgb/city/image.png
            # 而不是標準的: root/val/leftImg8bit/city/image.png

            img_folder_name = "rgb"  # 根據您的 ls 輸出

            # 判斷標籤資料夾名稱
            if self._n_classes == 19:
                lbl_folder_name = "labels_19" # 根據您的 ls 輸出
            else:
                lbl_folder_name = "labels_33"

            # 構建路徑
            self.images = [os.path.join(data_dir, folder_split, img_folder_name, p) for p in rgb_rel_paths]
            self.labels = [os.path.join(data_dir, folder_split, lbl_folder_name, p) for p in label_rel_paths]

            print(f"Loaded {len(self.images)} images and {len(self.labels)} labels.")

            if len(self.images) == 0:
                raise RuntimeError(f"No images found for split {split}! Check if {rgb_list_file} is empty.")

            assert len(self.images) == len(self.labels), "Mismatch between images and labels count!"

            self._files = {}

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        # class names, class colors
        if self._n_classes == 19:
            self._class_names = self.CLASS_NAMES_REDUCED
            self._class_colors = np.array(self.CLASS_COLORS_REDUCED, dtype="uint8")
        else:
            self._class_names = self.CLASS_NAMES_FULL
            self._class_colors = np.array(self.CLASS_COLORS_FULL, dtype="uint8")

    @property
    def cameras(self): return self._cameras
    @property
    def class_names(self): return self._class_names
    @property
    def class_names_without_void(self): return self._class_names[1:]
    @property
    def class_colors(self): return self._class_colors
    @property
    def class_colors_without_void(self): return self._class_colors[1:]
    @property
    def n_classes(self): return self._n_classes + 1
    @property
    def n_classes_without_void(self): return self._n_classes
    @property
    def split(self): return self._split
    @property
    def source_path(self): return os.path.abspath(os.path.dirname(__file__))
    @property
    def with_input_orig(self): return self._with_input_orig

    def _load(self, filename):
        if not os.path.exists(filename):
             # 增加更詳細的報錯，幫助 Debug
             raise FileNotFoundError(f"Image/Label file not found: {filename}")
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"Failed to load image (cv2 returned None): {filename}")
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_name(self, idx): return self.images[idx]
    def load_image(self, idx): return self._load(self.images[idx])
    def load_label(self, idx):
        # 讀取原始標籤
        label = self._load(self.labels[idx])

        # === 修正開始 ===
        # 將標籤轉換為 int32 以避免 overflow 問題
        label = label.astype(np.int32)

        # 處理 Void 類別：
        # 如果原始資料中有 19 (代表 void/ignore)，將其設為 -1
        # 這樣後面 +1 之後，它就會變成 0 (這是 Loss 函數通常忽略的 index)
        label[label == 19] = -1

        # 處理可能的 255 (標準 void)，也設為 -1
        label[label == 255] = -1

        # 整體 +1，使有效類別從 1 開始 (1-19)，Void 變為 0
        label = label + 1
        # === 修正結束 ===

        return label

    def __len__(self):
        if self.overfit: return 2
        return len(self.images)
