#!/bin/bash

echo "正在重設 PYTORCH_CUDA_ALLOC_CONF..."
unset PYTORCH_CUDA_ALLOC_CONF

IMG_PATH="./dataset_AnomalyTrack/images/elephant0000.jpg"
CKPT_PATH="/mnt/8tb_hdd2/ContMAV/results/cityscapes/Test_With_Pretrained2/30_11_2025-17_24_46-723110/best_miou.pth"
SAVE_PREFIX="elephant0000"

echo "Processing: $IMG_PATH"

python inference_test.py \
    --img_path "$IMG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --dataset_dir ./datasets/cityscapes \
    --batch_size 1 \
    --no_imagenet_pretraining \
    --encoder resnet34 \
    --height 512 --width 1024 \
    --save_seg_overlay "seg_overlay_${SAVE_PREFIX}.png" \
    --save_unk_overlay "unk_overlay_${SAVE_PREFIX}.png" \
    --save_unk_mask "unk_mask_${SAVE_PREFIX}.png"

echo "Done!"
echo "Seg overlay:   seg_overlay_${SAVE_PREFIX}.png"
echo "Unknown overlay: unk_overlay_${SAVE_PREFIX}.png"
echo "Unknown mask:  unk_mask_${SAVE_PREFIX}.png"
