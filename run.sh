#!/bin/bash

# 设置工作目录为脚本所在目录
cd "$(dirname "$0")"

echo "Starting improved training for BLEU 25-30..."

# 创建日志目录
mkdir -p logs

# 训练改进模型
echo "Training improved model..."
cd src
python train_improved.py --config ../configs/improved_25plus.yaml --gpu 0 --log_dir ../logs

echo "Training completed!"
echo "Check results in ../results/improved_25plus/"
echo "Check logs in ../logs/"