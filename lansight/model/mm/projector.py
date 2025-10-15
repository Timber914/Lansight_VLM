"""
Vision-to-text projector: map vision encoder hidden size to LM hidden size.
Kept minimal for clarity; can be replaced by a deeper MLP if needed.
"""
from __future__ import annotations
from torch import nn


class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size: int = 768, hidden_size: int = 512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(nn.Linear(self.ve_hidden_size, self.hidden_size))

    def forward(self, image_encoders):
        return self.vision_proj(image_encoders)


__all__ = ["VisionProj"]

