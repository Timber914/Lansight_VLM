"""
LanSight 预训练：学习图像-文本对齐（图像描述），冻结视觉编码器，仅训练投影等少量参数。
"""

import argparse
import time
import math
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch
import torch.distributed as dist

__package__ = "trainer"
from lansight.utils.paths import MODEL_DIR, VISION_MODEL_DIR, OUT_DIR, DATASETS_DIR, PRETRAIN_JSONL, PRETRAIN_IMAGES_DIR, RUNLOGS_DIR
from pathlib import Path as _P
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from lansight.utils.demo_tokenizer import BasicChatTokenizer
from lansight.model.lansight_vlm import LanSightVLM, VLMConfig
from lansight.dataset.vlm_dataset import VLMDataset


def Logger(content):
    """仅在主进程打印日志(DDP 场景友好)。"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """余弦退火学习率调度。"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """单个 epoch 训练循环：AMP + 梯度累积 + Clip Grad。"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)
        pixel_values = pixel_values.to(args.device, non_blocking=True)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if model_config.use_moe else ''
            global_step = epoch * iter_per_epoch + (step + 1)
            ckp = f"{args.save_dir}/pretrain_vlm_{model_config.hidden_size}{moe_path}_step{global_step:06d}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # 半精度保存
            torch.save(clean_state_dict, ckp)
            # 删除上一个检查点
            prev_step = global_step - args.save_interval
            if prev_step > 0:
                prev_ckp = f"{args.save_dir}/pretrain_vlm_{model_config.hidden_size}{moe_path}_step{prev_step:06d}.pth"
                try:
                    if os.path.exists(prev_ckp):
                        os.remove(prev_ckp)
                except Exception as e:
                    Logger(f"[WARN] 删除旧权重失败: {prev_ckp} -> {e}")
            # 自动导出 Transformers 结构（覆盖 out/transformers/LanSight_Model）
            try:
                from lansight.scripts.convert import convert_torch2transformers
                import torch as _torch
                tf_dir = _P(OUT_DIR) / 'transformers' / 'LanSight_Model'
                os.makedirs(tf_dir, exist_ok=True)
                prec_map = {'bf16': _torch.bfloat16, 'fp16': _torch.float16, 'fp32': _torch.float32}
                use_dtype = prec_map.get(str(getattr(args, 'precision', 'bf16')).lower(), _torch.bfloat16)
                Logger(f"[export] 导出 Transformers 到: {tf_dir}")
                convert_torch2transformers(ckp, str(tf_dir), model_config, dtype=use_dtype)
            except Exception as ex:
                Logger(f"[WARN] 自动导出 Transformers 失败: {ex}")
            model.train()

        # 可选：限制每轮的训练步数用于快速验证
        if hasattr(args, 'max_steps') and args.max_steps and (step + 1) >= args.max_steps:
            break


def init_model(model_config: VLMConfig):
    """初始化分词器与模型，按需冻结参数并统计可训练参数量。"""
    tokenizer = BasicChatTokenizer() if args.demo_tokenizer else AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=False)
    moe_path = '_moe' if model_config.use_moe else ''
    # 加载纯语言模型权重
    ckp = f'{args.save_dir}/llm_{model_config.hidden_size}{moe_path}.pth'
    model = LanSightVLM(model_config, vision_model_path=str(VISION_MODEL_DIR))
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 冻结除 vision_proj 外的所有参数
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    Logger(f'VLM可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    _, preprocess = model.vision_encoder, model.processor
    return model.to(args.device), tokenizer, preprocess


def setup_cuda():
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanSight-V Pretrain")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--fused_optimizer", action="store_true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--demo_tokenizer", action="store_true", help="使用内置 Demo 分词器以便在本机快速跑通")
    parser.add_argument("--wandb_project", type=str, default="LanSight-V")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=str(PRETRAIN_JSONL))
    # 使用预训练图片目录（本地已解压）：dataset/pretrain_images/pretrain_images
    parser.add_argument("--images_path", type=str, default=str(PRETRAIN_IMAGES_DIR))
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0, help="每轮最多训练多少步，0 表示不限制")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=640, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    args = parser.parse_args()

    model_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             max_seq_len=args.max_seq_len)
    max_seq_len = model_config.max_seq_len
    # 统一将模型权重保存到 runlogs/
    args.save_dir = str(RUNLOGS_DIR)
    os.makedirs(RUNLOGS_DIR, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"LanSight-V Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 自动精度：GPU 不支持 bf16 时回退到 fp16
    precision_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': None}
    amp_dtype = precision_map.get(args.precision, torch.bfloat16)
    if device_type == 'cuda' and args.precision in ('bf16', 'bfloat16'):
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = (0, 0)
        if major < 8:
            print('[warn] CUDA 不支持 bfloat16，自动切换到 fp16')
            args.precision = 'fp16'
            amp_dtype = torch.float16
    ctx = nullcontext() if (device_type=='cpu' or args.precision=='fp32') else torch.cuda.amp.autocast(dtype=amp_dtype)
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model_config.gradient_checkpointing = args.grad_checkpoint
    model, tokenizer, preprocess = init_model(model_config)

    setup_cuda()
    if args.compile and device_type=="cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            Logger(f"torch.compile 未启用: {e}")

    print(f"[pretrain] data_path: {args.data_path}")
    print(f"[pretrain] images_path: {args.images_path}")
    train_ds = VLMDataset(args.data_path, args.images_path, tokenizer, preprocess=preprocess,
                          image_special_token=model_config.image_special_token,
                          max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        prefetch_factor=args.prefetch_factor if args.num_workers>0 else None,
        persistent_workers=args.persistent_workers if args.num_workers>0 else False
    )

    use_scaler = (args.precision=='fp16' and device_type=='cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    # fused AdamW if available
    fused_kw = {}
    try:
        if args.fused_optimizer and device_type=='cuda':
            import inspect as _insp
            if 'fused' in _insp.signature(optim.AdamW).parameters:
                fused_kw['fused'] = True
    except Exception:
        fused_kw = {}
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, **fused_kw)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        ddp_kwargs = dict(device_ids=[ddp_local_rank], find_unused_parameters=False)
        try:
            ddp_kwargs['static_graph'] = True
        except Exception:
            pass
        model = DistributedDataParallel(model, **ddp_kwargs)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)
