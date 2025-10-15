"""
LanSightVLM: vision-language wrapper that injects CLIP features into the LM.
This module depends on the language model backbone and separates vision utils.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Union, Dict

import torch
from torch import nn

from ..model_lansight import LanSightForCausalLM, MOEFeedForward
from ..config import VLMConfig
from ..vision.clip import load_clip, image2tensor
from .projector import VisionProj


class LanSightVLM(LanSightForCausalLM):
    """LanSight VLM: freeze vision encoder, project features and fuse into LM tokens.
    The image location is detected via repeated image_ids pattern; fused once at start.
    """
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path: str = "./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params or VLMConfig())
        self.params: VLMConfig = params or VLMConfig()
        self.vision_encoder, self.processor = load_clip(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=self.params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        return load_clip(model_path)

    @staticmethod
    def image2tensor(image, processor):
        return image2tensor(image, processor)

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def _find_image_spans(self, tokens: torch.Tensor, image_ids: List[int]) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        image_ids_tensor = torch.tensor(image_ids, device=tokens.device)
        L = len(image_ids)
        if L > tokens.size(1):
            return None
        # rolling window equality check
        tokens_view = tokens.unfold(1, L, 1)
        matches = (tokens_view == image_ids_tensor).all(dim=2)
        spans: Dict[int, List[Tuple[int, int]]] = {}
        for b in range(tokens.size(0)):
            idxs = matches[b].nonzero(as_tuple=True)[0]
            if idxs.numel():
                spans[b] = [(i.item(), i.item() + L - 1) for i in idxs]
        return spans or None

    def count_vision_proj(self, tokens: torch.Tensor, h: torch.Tensor, vision_tensors=None, seqlen: int = 512):
        spans = self._find_image_spans(tokens, self.params.image_ids)
        if vision_tensors is None or not spans:
            return h
        vision_proj = self.vision_proj(vision_tensors)
        if len(vision_proj.shape) == 3:
            vision_proj = vision_proj.unsqueeze(0)
        new_h = []
        for i in range(h.size(0)):
            if i in spans:
                h_i = h[i]
                img_idx = 0
                for start_idx, end_idx in spans[i]:
                    if img_idx < vision_proj.size(1):
                        h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[:seqlen]
                        img_idx += 1
                new_h.append(h_i)
            else:
                new_h.append(h[i])
        return torch.stack(new_h, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        **args,
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack(
                [self.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder) for i in range(num)],
                dim=stack_dim,
            )
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss for layer in self.model.layers if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


__all__ = ["LanSightVLM", "VLMConfig"]

