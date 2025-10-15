"""
Compatibility shim: keep old import path working.
"""
from .mm.vlm import LanSightVLM
from .config import VLMConfig

__all__ = ["LanSightVLM", "VLMConfig"]
