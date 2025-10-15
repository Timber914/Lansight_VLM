import os
import sys
import argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from lansight.model.lansight_vlm import LanSightVLM, VLMConfig

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(torch_path, transformers_path, lm_config: VLMConfig, dtype=torch.bfloat16):
    """将原生 .pth 权重转换为 Transformers 目录结构。"""
    VLMConfig.register_for_auto_class()
    LanSightVLM.register_for_auto_class("AutoModelForCausalLM")

    root = __import__('pathlib').Path(__file__).resolve().parents[2]
    vision_dir = root / 'lansight' / 'model' / 'vision_model' / 'clip-vit-base-patch16'
    lm_model = LanSightVLM(lm_config, vision_model_path=str(vision_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6:.3f} 百万 ({model_params / 1e9:.3f} B)')

    # 移除视觉编码器（保持 transformers 侧纯文本权重；推理时再注入 vision_encoder）
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)

    tokenizer_dir = root / 'lansight' / 'model'
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-LanSight-V 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LanSight 权重格式转换')
    parser.add_argument('--mode', type=str, default='sft', choices=['sft', 'pretrain'], help='选择转换的权重类型')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', action='store_true', default=False)
    parser.add_argument('--max_seq_len', type=int, default=8192)
    parser.add_argument('--torch_path', type=str, default='')
    parser.add_argument('--from_out', type=str, default=str((__import__('pathlib').Path(__file__).resolve().parents[2] / 'out')),
                        help='原生权重所在目录（默认使用项目 out/）')
    parser.add_argument('--out_dir', type=str, default=str((__import__('pathlib').Path(__file__).resolve().parents[2] / 'out' / 'transformers' / 'LanSight_Model')),
                        help='Transformers 输出目录（默认写入项目 out/transformers/LanSight_Model）')
    parser.add_argument('--auto_latest', action='store_true', default=False,
                        help='自动选择 from_out 下最新的 step 权重文件（优先），找不到则回退到固定名')
    args = parser.parse_args()

    lm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                          max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    torch_path = args.torch_path
    if not torch_path:
        moe_path = '_moe' if args.use_moe else ''
        prefix = f"{args.mode}_vlm"
        from pathlib import Path as _P
        outp = _P(args.from_out)
        if args.auto_latest:
            patterns = [f"{prefix}_{args.hidden_size}{moe_path}_step*.pth", f"{prefix}_{args.hidden_size}{moe_path}.pth"]
            found = []
            for pat in patterns:
                found = sorted(outp.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
                if found:
                    break
            if not found:
                torch_path = str(outp / f"{prefix}_{args.hidden_size}{moe_path}.pth")
            else:
                torch_path = str(found[0])
        else:
            torch_path = str(outp / f"{prefix}_{args.hidden_size}{moe_path}.pth")

    print(f"[convert] 使用原生权重: {torch_path}")
    transformers_path = args.out_dir
    convert_torch2transformers(torch_path, transformers_path, lm_config)
