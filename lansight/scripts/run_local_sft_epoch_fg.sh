#!/usr/bin/env bash
# 在本机前台运行 1 个 epoch 的 SFT 训练（实时在当前终端显示日志）
# 保存：每 200 步保存一次，并删除上一个检查点（由训练脚本实现）
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[ERROR] 虚拟环境不存在：$VENV" >&2
  exit 1
fi
source "$VENV/bin/activate"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
python -u -m lansight.trainer.train_sft \
  --device cpu \
  --epochs 1 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --num_workers 0 \
  --data_path datasets/sft_data.jsonl \
  --images_path datasets/sft_images/sft_images \
  --log_interval 20 \
  --save_interval 200 \
  --demo_tokenizer
