"""
一键启动训练脚本：直接运行本文件即可开始训练。
按需修改下面的配置项，每一项一行，右侧注释说明作用与可选值。
"""
import os
import sys
import shlex
import subprocess
from datetime import datetime

# 训练模式：'pretrain' 预训练（图文对齐），'sft' 微调（看图对话）
MODE = 'pretrain'  # 'pretrain' 或 'sft'

# 设备：'cuda:0' 使用首块 GPU；无 GPU 可改为 'cpu'
DEVICE = 'cuda:0'

# 轮数：训练多少个 epoch（整数）
EPOCHS = 1

# 批量：单次迭代的样本数（根据显存调整）
BATCH_SIZE = 16

# 精度：'bf16'、'fp16'、'fp32'（脚本会自动降级不支持的 bf16 到 fp16）
PRECISION = 'bf16'

# 数据工作线程：DataLoader 线程数；受环境限制可设为 0 避免卡住
NUM_WORKERS = 4

# 最多步数：每轮最多训练多少步（0 表示不限制；先小跑可设为 100）
MAX_STEPS = 0

# 日志间隔：每隔多少步打印一次日志（整数）
LOG_INTERVAL = 50

# 保存间隔：每隔多少步保存一次权重（默认 500，训练脚本已默认）
SAVE_INTERVAL = 500

# 文本/图像路径：不填则使用项目内默认（datasets/ 下），也可显式指定
DATA_PATH = ''  # 例：'datasets/pretrain_data.jsonl' 或 'datasets/sft_data.jsonl'
IMAGES_PATH = ''  # 例：'datasets/pretrain_images' 或 'datasets/sft_images'

# 模型结构：隐藏维度与层数（与权重/显存匹配）
HIDDEN_SIZE = 512  # 512 或 768
NUM_HIDDEN_LAYERS = 8  # 512→8 层；768→16 层
MAX_SEQ_LEN = 640  # 最大序列长度（预训练可用较小值以加快速度）

# 其它开关：是否启用 grad checkpoint 与 torch.compile（可能提速）
GRAD_CHECKPOINT = False
USE_COMPILE = False

# Demo 分词器：True 时使用内置简化分词器（便于无 tokenizers 环境快速跑通）
DEMO_TOKENIZER = False


def main() -> int:
    root = os.path.abspath(os.path.dirname(__file__))
    # 统一在 runlogs 下保存权重与日志（训练脚本已改为保存到 runlogs）
    runlogs = os.path.join(root, 'runlogs')
    os.makedirs(runlogs, exist_ok=True)

    if MODE not in ('pretrain', 'sft'):
        print(f"[ERR] MODE 不合法: {MODE}")
        return 1

    # 选择训练模块
    module = 'lansight.trainer.train_pretrain' if MODE == 'pretrain' else 'lansight.trainer.train_sft'

    # 构造命令
    cmd = [sys.executable, '-u', '-m', module,
           '--device', DEVICE,
           '--epochs', str(EPOCHS),
           '--batch_size', str(BATCH_SIZE),
           '--precision', PRECISION,
           '--num_workers', str(NUM_WORKERS),
           '--log_interval', str(LOG_INTERVAL),
           '--save_interval', str(SAVE_INTERVAL),
           '--hidden_size', str(HIDDEN_SIZE),
           '--num_hidden_layers', str(NUM_HIDDEN_LAYERS),
           '--max_seq_len', str(MAX_SEQ_LEN),
          ]

    if MAX_STEPS and int(MAX_STEPS) > 0:
        cmd += ['--max_steps', str(MAX_STEPS)]
    if GRAD_CHECKPOINT:
        cmd += ['--grad_checkpoint']
    if USE_COMPILE:
        cmd += ['--compile']
    if DEMO_TOKENIZER:
        cmd += ['--demo_tokenizer']

    # 路径参数（不传则训练脚本使用项目内默认 datasets/）
    if DATA_PATH:
        cmd += ['--data_path', DATA_PATH]
    if IMAGES_PATH:
        cmd += ['--images_path', IMAGES_PATH]

    # 日志文件（时间戳）
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = f"{MODE}_epoch{EPOCHS}_bs{BATCH_SIZE}_{ts}.log"
    log_path = os.path.join(runlogs, log_name)

    print('[start] 执行命令:')
    print(' '.join(shlex.quote(x) for x in cmd))
    print(f"[start] 日志文件: {log_path}")

    # 运行并实时写日志
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True) as p, \
            open(log_path, 'w', encoding='utf-8') as lf:
        for line in p.stdout:  # type: ignore[attr-defined]
            sys.stdout.write(line)
            lf.write(line)
        p.wait()
        return p.returncode or 0


if __name__ == '__main__':
    raise SystemExit(main())

