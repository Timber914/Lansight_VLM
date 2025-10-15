"""
一个最小可用的分词器用于 Demo：
- 提供与 Transformers Tokenizer 类似的接口：from_pretrained、__call__、decode、apply_chat_template
- 词表按需增长，Token id 范围控制在 [0, vocab_size)
- 特殊约定：'@' 字符映射为 token id=34；连续 196 个'@'可表示 1 张图片占位符
注意：仅用于在当前机器上跑通 Demo，非真实训练/推理用分词器。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


class _Batch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self
    def __getitem__(self, key):
        if key == 'input_ids':
            return self.input_ids
        if key == 'attention_mask':
            return self.attention_mask
        raise KeyError(key)


class BasicChatTokenizer:
    def __init__(self, vocab_size: int = 6400):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._next_id = 100  # 预留 [0..99] 给特殊用途
        self._vocab: Dict[str, int] = {
            '<pad>': self.pad_token_id,
            '<eos>': self.eos_token_id,
            '@': 34,
        }
        self._inv_vocab: Dict[int, str] = {v: k for k, v in self._vocab.items()}

    @classmethod
    def from_pretrained(cls, path: str, use_fast: bool = False):
        return cls()

    def _token_id(self, token: str) -> int:
        if token in self._vocab:
            return self._vocab[token]
        # 分配新 id，确保不越界
        tid = self._next_id
        self._next_id = min(self.vocab_size - 1, self._next_id + 1)
        self._vocab[token] = tid
        self._inv_vocab[tid] = token
        return tid

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True):
        parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            parts.append(f"{role}: {content}\n")
        if add_generation_prompt:
            parts.append('assistant: ')
        text = ''.join(parts)
        if not tokenize:
            return text
        return self(text)

    def __call__(self, text: str, return_tensors: str = None, truncation: bool = True, **kwargs):
        # 将连续的'@'转换为等长的 34
        ids: List[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '@':
                j = i
                while j < len(text) and text[j] == '@':
                    ids.append(34)
                    j += 1
                i = j
                continue
            # 简单按空白分词：抓取连续非空白作为 token
            if ch.isspace():
                i += 1
                continue
            j = i + 1
            while j < len(text) and (not text[j].isspace()) and text[j] != '@':
                j += 1
            token = text[i:j]
            ids.append(self._token_id(token))
            i = j

        # 截断保护
        if truncation and len(ids) > 8192:
            ids = ids[-8192:]

        # 伪造 attention_mask
        attn = [1] * len(ids)
        if return_tensors == 'pt':
            import torch
            return _Batch(input_ids=torch.tensor([ids], dtype=torch.long),
                          attention_mask=torch.tensor([attn], dtype=torch.long))
        else:
            class _Enc:
                pass
            enc = _Enc()
            enc.input_ids = ids
            enc.attention_mask = attn
            return enc

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if isinstance(i, list):
                i = i[0]
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self._inv_vocab.get(int(i), '▯'))
        return ' '.join(out)
