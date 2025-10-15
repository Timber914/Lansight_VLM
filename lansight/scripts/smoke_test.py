"""
LanSight 快速冒烟测试：
- 加载本地权重（sft/pretrain 二选一），构造包含图像占位符的假输入，跑一次前向。
- 用于验证本机依赖与模型/视觉编码器是否可正常工作（无需分词器）。
"""
import os
import torch
from PIL import Image
from pathlib import Path
from lansight.model.lansight_vlm import LanSightVLM, VLMConfig


def main():
    root = Path(__file__).resolve().parents[2]
    device = torch.device('cpu')

    # 配置与模型
    cfg = VLMConfig(hidden_size=512, num_hidden_layers=8, max_seq_len=512)
    model = LanSightVLM(cfg, vision_model_path=str(root/'lansight'/'model'/'vision_model'/'clip-vit-base-patch16'))

    # 加载本地权重（若存在 SFT 则优先）；优先使用 out/ 目录，避免依赖 assets/torch_weights
    wt_dir = root/'out'
    for name in ['sft_vlm_512.pth', 'pretrain_vlm_512.pth']:
        p = wt_dir/name
        if p.exists():
            sd = torch.load(p, map_location=device)
            model.load_state_dict(sd, strict=False)
            print(f'Loaded weights: {p.name}')
            break
    model.eval().to(device)

    # 构造假输入（含 196 个图像占位符 token id=34）
    seq_len = 320
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(1, seq_len), dtype=torch.long)
    start = 50
    image_ids = torch.tensor(cfg.image_ids, dtype=torch.long).unsqueeze(0)
    input_ids[:, start:start+196] = image_ids

    # 准备一张示例图片（优先 datasets）
    from lansight.utils.paths import DATASETS_DIR
    cand = [
        DATASETS_DIR/'eval_images',
        DATASETS_DIR/'sft_images',
        DATASETS_DIR/'sft_images'/'sft_images',
        DATASETS_DIR/'pretrain_images',
        DATASETS_DIR/'pretrain_images'/'pretrain_images',
    ]
    img_dir = next((p for p in cand if p.exists()), None)
    assert img_dir is not None, '未找到示例图片目录（datasets/eval_images 或 sft_images[/sft_images] 或 pretrain_images[/pretrain_images]）'
    img_files = list(img_dir.glob('*.jpg'))
    assert img_files, f'目录下没有jpg图片: {img_dir}'
    image = Image.open(img_files[0]).convert('RGB')
    pixel = LanSightVLM.image2tensor(image, model.processor).to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), pixel_values=pixel)
    print('logits shape:', tuple(out.logits.shape))
    print('OK')


if __name__ == '__main__':
    main()
