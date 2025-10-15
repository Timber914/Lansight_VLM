#!/usr/bin/env bash
# 在本机（CPU）跑 SFT 1 个 epoch（使用官方 sft_data.jsonl + 解压的 sft_images）
# 注：CPU 训练极慢，整个 epoch 可能需要数十小时；建议后台运行并监控日志。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT_DIR/.venv"
LOG_DIR="$ROOT_DIR/runlogs"
mkdir -p "$LOG_DIR"
if [ ! -d "$VENV" ]; then
  echo "[ERROR] 虚拟环境不存在：$VENV" >&2
  exit 1
fi
source "$VENV/bin/activate"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
CMD=(python -m lansight.trainer.train_sft
  --device cpu
  --epochs 1
  --batch_size 1
  --learning_rate 1e-5
  --num_workers 0
  --data_path datasets/sft_data.jsonl
  --images_path datasets/sft_images/sft_images
  --log_interval 20
  --save_interval 200
  --demo_tokenizer
)
TS="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$LOG_DIR/sft_epoch1_$TS.log"
echo "[INFO] 启动训练，日志：$LOGFILE"
nohup "${CMD[@]}" > "$LOGFILE" 2>&1 &
echo $! > "$LOG_DIR/sft_epoch1_$TS.pid"
echo "[INFO] 进程 PID: $(cat "$LOG_DIR/sft_epoch1_$TS.pid")"
echo "[HINT] 监控日志：tail -f '$LOGFILE'"
