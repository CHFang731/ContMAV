#!/bin/bash

echo "正在重設 PYTORCH_CUDA_ALLOC_CONF..."
unset PYTORCH_CUDA_ALLOC_CONF

echo "開始執行訓練 (使用 ImageNet 預訓練)..."

# 注意：
# 1. 移除了 --no_imagenet_pretraining (這樣就會預設為 True)
# 2. 移除了 --load_weights (因為我們是讓 Encoder 自己去載入)
# 3. 確保 --pretrained_dir 指向我們剛建立的資料夾 (預設就是 ./trained_models/imagenet，所以其實可以不寫，但寫出來比較清楚)

python train.py \
    --dataset_dir /mnt/8tb_hdd2/ContMAV/datasets/cityscapes \
    --batch_size 8 \
    --epochs 400 \
    --pretrained_dir ./trained_models/imagenet \
    --id Test_With_Pretrained2 \
    

echo "訓練程序已結束。"
