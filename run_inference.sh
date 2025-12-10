#!/bin/bash

# 1. 解決 PyTorch 報錯
echo "正在重設 PYTORCH_CUDA_ALLOC_CONF..."
unset PYTORCH_CUDA_ALLOC_CONF

# 2. 設定參數 (您可以在這裡修改圖片路徑)
IMG_PATH="./dataset_AnomalyTrack/images/elephant0000.jpg"
CKPT_PATH="/mnt/8tb_hdd2/ContMAV/results/cityscapes/Test_With_Pretrained2/30_11_2025-17_24_46-723110/best_miou.pth"
SAVE_PATH="result_elephant.png"

echo "Processing: $IMG_PATH"

# 3. 執行推論
python inference.py \
    --img_path "$IMG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --save_path "$SAVE_PATH" \
    --dataset_dir ./datasets/cityscapes \
    --batch_size 1 \
    --no_imagenet_pretraining

echo "Done! Result saved to $SAVE_PATH"
