"""
Checkpoint helpers for loading/saving LanSight weights.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import os
import torch


def load_state_dict_safe(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a state_dict from file if present; return empty dict if missing.
    Useful to allow running without local weights.
    """
    if not path or (not os.path.exists(path)):
        return {}
    return torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))


def strip_vision_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove vision_encoder.* entries from a VLM state_dict."""
    return {k: v for k, v in state.items() if not k.startswith("vision_encoder.")}


def save_pytorch(path: str, state: Dict[str, Any], half: bool = True):
    """Save PyTorch weights, optionally converting tensors to half precision."""
    if half:
        state = {k: (v.half() if hasattr(v, "half") else v) for k, v in state.items()}
    torch.save(state, path)


__all__ = ["load_state_dict_safe", "strip_vision_prefix", "save_pytorch"]

