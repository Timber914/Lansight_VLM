"""
CLIP vision encoder utilities: load/freeze model and preprocess images.
Separated from VLM to decouple vision dependency from LM.
"""
from __future__ import annotations
import os
from typing import Tuple, Optional

from transformers import CLIPProcessor, CLIPModel


def load_clip(model_path: str) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """Load a local CLIP model + processor and freeze model params.
    Returns (None, None) if path is missing to allow CPU-only demos to proceed.
    """
    if not os.path.exists(model_path):
        return None, None
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval(), processor


def image2tensor(image, processor):
    """Convert PIL image to CLIP pixel_values tensor (BCHW without batch)."""
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    return processor(images=image, return_tensors="pt")["pixel_values"]


__all__ = ["load_clip", "image2tensor"]

