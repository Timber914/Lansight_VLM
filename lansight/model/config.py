"""
Configuration objects for LanSight: language-only and VLM (vision-language) variants.
Keep surface compatible with existing imports by re-exporting LanSightConfig.
"""
from __future__ import annotations
from typing import List

# Reuse the proven LM config implementation
from .model_lansight import LanSightConfig as _LanSightConfig


class LanSightConfig(_LanSightConfig):
    """Alias to the language model config (kept for explicit import path)."""
    pass


class VLMConfig(LanSightConfig):
    """VLM config extends LanSightConfig by adding image placeholder settings."""
    model_type = "lansight-v"

    def __init__(
        self,
        image_special_token: str = '@' * 196,
        image_ids: List[int] = None,
        **kwargs,
    ):
        if image_ids is None:
            image_ids = [34] * 196
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)


__all__ = ["LanSightConfig", "VLMConfig"]

