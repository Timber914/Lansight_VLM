# -*- coding: utf-8 -*-
"""
路径工具：统一管理项目内的关键目录，避免硬编码。
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
LAN_DIR = ROOT / 'lansight'
MODEL_DIR = LAN_DIR / 'model'
VISION_MODEL_DIR = MODEL_DIR / 'vision_model' / 'clip-vit-base-patch16'
ASSETS_DIR = ROOT / 'assets'  # kept for backward-compat; may not exist
# Prefer plural 'datasets'; fall back to singular 'dataset' if needed
DATASETS_DIR = ROOT / 'datasets'
if not DATASETS_DIR.exists():
    _alt = ROOT / 'dataset'
    if _alt.exists():
        DATASETS_DIR = _alt
# default nested layout in provided demo packs (support nested fallback)
IMAGES_DIR = DATASETS_DIR / 'eval_images'
TRANSFORMERS_DIR = ROOT / 'out' / 'transformers'
TORCH_WEIGHTS_DIR = ROOT / 'out'
OUT_DIR = ROOT / 'out'
RUNLOGS_DIR = ROOT / 'runlogs'
PRETRAIN_JSONL = DATASETS_DIR / 'pretrain_data.jsonl'
SFT_JSONL = DATASETS_DIR / 'sft_data.jsonl'
# nested folder as shipped: datasets/sft_images/sft_images
from pathlib import Path as _P
# Prefer flat layout (datasets/sft_images) but accept nested (datasets/sft_images/sft_images)
_sft_cands = [DATASETS_DIR / 'sft_images', DATASETS_DIR / 'sft_images' / 'sft_images']
SFT_IMAGES_DIR = next((p for p in _sft_cands if p.exists()), _sft_cands[0])
_pt_cands = [DATASETS_DIR / 'pretrain_images', DATASETS_DIR / 'pretrain_images' / 'pretrain_images']
PRETRAIN_IMAGES_DIR = next((p for p in _pt_cands if p.exists()), _pt_cands[0])
