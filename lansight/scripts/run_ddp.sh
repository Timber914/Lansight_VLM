#!/usr/bin/env bash
# 用法示例：在 2 张 GPU 上运行 SFT 训练
# torchrun --nproc_per_node=2 lansight/trainer/train_sft.py --epochs 1 --batch_size 4 --precision bf16 --grad_checkpoint --compile
