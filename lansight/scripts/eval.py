"""
LanSight 推理/聊天脚本：支持原生权重或本地 Transformers 权重加载；
提供单图/多图两种推理模式与流式输出。
"""

import argparse
import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from lansight.utils.paths import VISION_MODEL_DIR, DATASETS_DIR
from lansight.utils.demo_tokenizer import BasicChatTokenizer
from lansight.model.lansight_vlm import LanSightVLM, VLMConfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device):
    # 按加载模式选择分词器来源
    if args.load == 0:
        tokenizer = BasicChatTokenizer() if args.demo_tokenizer else AutoTokenizer.from_pretrained(str((__import__('pathlib').Path(__file__).resolve().parents[2] / 'lansight' / 'model')), use_fast=False)
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain_vlm', 1: 'sft_vlm', 2: 'sft_vlm'}
        # 自动选择最新 step 权重（优先），找不到则回退到固定名
        if args.auto_latest:
            from pathlib import Path as _P
            outp = _P(args.out_dir)
            pats = [f"{modes[args.model_mode]}_{args.hidden_size}{moe_path}_step*.pth", f"{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth"]
            found = []
            for pat in pats:
                found = sorted(outp.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
                if found:
                    break
            if not found:
                ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'
            else:
                ckp = str(found[0])
        else:
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'
        print(f"[eval] 使用权重: {ckp}")
        # 优先使用项目内置的视觉编码器目录，避免依赖 assets/*
        model = LanSightVLM(lm_config, vision_model_path=str(VISION_MODEL_DIR))
        # 若找不到权重文件，则以随机初始化运行（便于在无 assets/torch_weights 时仍可跑通）
        try:
            state_dict = torch.load(ckp, map_location=device)
            model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
        except FileNotFoundError:
            print(f"[eval][warn] 未发现本地权重: {ckp}，将以随机初始化权重运行。")
    else:
        # 支持从 --pretrained 指定的路径/仓库加载；否则默认使用 out/transformers/LanSight2-V
        default_tf_dir = __import__('pathlib').Path(__file__).resolve().parents[2] / 'out' / 'transformers' / 'LanSight2-V'
        transformers_src = args.pretrained or str(default_tf_dir)
        print(f"[eval] 使用 Transformers 源: {transformers_src}")
        tokenizer = BasicChatTokenizer() if args.demo_tokenizer else AutoTokenizer.from_pretrained(transformers_src, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(transformers_src, trust_remote_code=True)
        # 注入视觉编码器（优先本地项目内置目录）
        model.vision_encoder, model.processor = LanSightVLM.get_vision_model(str(VISION_MODEL_DIR))

    print(f'VLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    vision_model, preprocess = model.vision_encoder, model.processor
    return model.eval().to(device), tokenizer, vision_model.eval().to(device), preprocess


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with LanSight")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.65, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # LanSight2-Small (26M)：(hidden_size=512, num_hidden_layers=8)
    # LanSight2 (104M)：(hidden_size=768, num_hidden_layers=16)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 默认单图推理，设置为2为多图推理
    parser.add_argument('--use_multi', default=1, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--auto_latest', action='store_true', default=False,
                        help='自动选择 out_dir 下最新的 step 权重文件（仅在 --load 0 时生效）')
    parser.add_argument('--demo_tokenizer', default=False, action='store_true',
                        help='使用内置简化分词器以便在本机环境快速运行（仅用于Demo）')
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--pretrained', default=os.environ.get('LAN_PRETRAINED', ''), type=str,
                        help='当 --load 1 时，Transformers 模型的本地目录或 HF 仓库名（默认使用 out/transformers/LanSight2-V）')
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: Pretrain模型，1: SFT模型，2: SFT-多图模型 (beta拓展)")
    args = parser.parse_args()

    lm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                          max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    model, tokenizer, vision_model, preprocess = init_model(lm_config, args.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    def _demo_generate(model, input_ids, attention_mask, pixel_values, max_new_tokens, eos_id, temperature=1.0, top_p=0.9):
        import torch
        ids = input_ids.clone()
        produced = 0
        for _ in range(max_new_tokens):
            out = model(input_ids=ids, attention_mask=attention_mask, pixel_values=pixel_values)
            logits = out.logits[:, -1, :]
            if temperature and temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # 在前几步禁止采样 EOS，避免空回复
            if produced < 3:
                probs[:, int(eos_id)] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            # top-p nucleus sampling（简化实现）
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf <= top_p
            mask[..., 0] = True
            filtered_idx = sorted_idx[0, mask[0]]
            filtered_probs = sorted_probs[0, mask[0]]
            filtered_probs = filtered_probs / filtered_probs.sum()
            next_id = filtered_idx[torch.multinomial(filtered_probs, num_samples=1)]
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            produced += 1
            if int(next_id) == int(eos_id):
                break
        return ids

    def chat_with_vlm(prompt, pixel_values, image_names):
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print(f'[Image]: {image_names}')
        print('🤖️: ', end='')
        if args.demo_tokenizer:
            generated_ids = _demo_generate(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                pixel_values,
                max_new_tokens=min(64, args.max_seq_len),
                eos_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            generated_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_seq_len,
                num_return_sequences=1,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                top_p=args.top_p,
                temperature=args.temperature,
                pixel_values=pixel_values
            )

        gen_tail = generated_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tail, skip_special_tokens=True)
        if args.demo_tokenizer and (not response.strip()):
            # Demo 模式下若解码为空，则打印部分原始 token id 以便观察
            try:
                toks = gen_tail.tolist()
            except Exception:
                toks = []
            response = '[ids] ' + ' '.join(str(x) for x in toks[:32])
        messages.append({"role": "assistant", "content": response})
        if args.demo_tokenizer:
            print(response, end='')
        print('\n\n')


    # 单图推理：每1个图像单独推理
    if args.use_multi == 1:
        # prefer datasets/eval_images, then flat/nested sft_images or pretrain_images
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[2]
        cand = [
            DATASETS_DIR / 'eval_images',
            DATASETS_DIR / 'sft_images',
            DATASETS_DIR / 'sft_images' / 'sft_images',
            DATASETS_DIR / 'pretrain_images',
            DATASETS_DIR / 'pretrain_images' / 'pretrain_images',
        ]
        found = next((str(p) for p in cand if p.exists()), None)
        if found is None:
            raise FileNotFoundError('未找到可用的图像目录（datasets/eval_images 或 sft_images[/sft_images] 或 pretrain_images[/pretrain_images]）')
        image_dir = found.rstrip('/') + '/'
        prompt = f"{model.params.image_special_token}\n描述一下这个图像的内容。"

        for image_file in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
            pixel_tensors = LanSightVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)
            chat_with_vlm(prompt, pixel_tensors, image_file)

    # 2图推理：目录下的两个图像编码，一次性推理（power by ）
    if args.use_multi == 2:
        args.model_mode = 2
        from pathlib import Path as _P
        # try a multi-image folder under datasets if provided (fallback to sft_images)
        pref = DATASETS_DIR / 'eval_multi_images' / 'bird'
        if not pref.exists():
            pref = DATASETS_DIR / 'sft_images'
        image_dir = str(pref) + '/'
        prompt = (f"{lm_config.image_special_token}\n"
                  f"{lm_config.image_special_token}\n"
                  f"比较一下两张图像的异同点。")
        pixel_tensors_multi = []
        for image_file in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
            pixel_tensors_multi.append(LanSightVLM.image2tensor(image, preprocess))
        pixel_tensors = torch.cat(pixel_tensors_multi, dim=0).to(args.device).unsqueeze(0)
        # 同样内容重复10次
        for _ in range(10):
            chat_with_vlm(prompt, pixel_tensors, (', '.join(os.listdir(image_dir))))
