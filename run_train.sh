#!/bin/bash

echo "正在重設 PYTORCH_CUDA_ALLOC_CONF..."
unset PYTORCH_CUDA_ALLOC_CONF

echo "開始執行訓練 (使用 ImageNet 預訓練 + 開啟 obj/mav/closs)..."

python train.py \
    --dataset cityscapes \
    --dataset_dir /mnt/8tb_hdd2/ContMAV/datasets/cityscapes \
    --batch_size 8 \
    --epochs 400 \
    --encoder resnet34 \
    --pretrained_dir ./trained_models/imagenet \
    --id Test_With_MAV \
    --obj True \
    --mav True \
    --closs True

echo "訓練程序已結束。"
