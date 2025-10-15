#!/usr/bin/env bash
# LanSight 一键 Demo 脚本（本机/CPU 友好）
# 作用：
# 1) 激活本项目虚拟环境（若不存在则提示）
# 2) 运行冒烟测试（不依赖分词器）
# 3) 运行 Demo 推理（使用内置分词器与简化采样逻辑）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "[ERROR] 虚拟环境不存在：$VENV" >&2
  echo "请先运行：python3 -m venv $VENV && source $VENV/bin/activate && pip install torch transformers pillow" >&2
  exit 1
fi

source "$VENV/bin/activate"
cd "$ROOT_DIR"

echo "[1/2] 冒烟测试..."
python -m lansight.scripts.smoke_test

echo "[2/2] Demo 推理（CPU + 内置分词器）..."
python -m lansight.scripts.eval \
  --device cpu \
  --out_dir out \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 32 \
  --use_multi 1 \
  --load 0 \
  --demo_tokenizer

echo "[DONE] Demo 完成"
