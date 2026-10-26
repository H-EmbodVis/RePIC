"""TensorRT-lowerable action denoiser body used by the RoboTwin Flash backend."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class TensorRTActionCore(nn.Module):
    """Action Transformer body with real-valued RoPE and explicit attention."""

    def __init__(self, action_expert: nn.Module, weight_dtype: torch.dtype | None = None):
        super().__init__()
        self.action_expert = copy.deepcopy(action_expert).to(dtype=weight_dtype) if weight_dtype else action_expert
        self.layers = len(action_expert.blocks)
        self.num_heads = action_expert.num_heads
        self.head_dim = action_expert.attn_head_dim
        freqs = action_expert.freqs[:32]
        self.register_buffer("rope_cos", freqs.real.view(1, 32, 1, -1).float(), persistent=False)
        self.register_buffer("rope_sin", freqs.imag.view(1, 32, 1, -1).float(), persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        pairs = x.unflatten(-1, (self.num_heads, self.head_dim // 2, 2)).float()
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * self.rope_cos - odd * self.rope_sin,
                            even * self.rope_sin + odd * self.rope_cos), dim=-1).flatten(2).to(x.dtype)

    def _attention(self, q, k, v, mask):
        q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.where(mask, scores, torch.full_like(scores, -65504.0))
        return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).flatten(2)

    def forward(self, action_tokens, t_mod, context, context_mask, action_mask, *kv):
        x = action_tokens
        for i, block in enumerate(self.action_expert.blocks):
            modulation = block.modulation + t_mod
            shift_msa, scale_msa, gate_msa = modulation[:, 0:1], modulation[:, 1:2], modulation[:, 2:3]
            shift_mlp, scale_mlp, gate_mlp = modulation[:, 3:4], modulation[:, 4:5], modulation[:, 5:6]
            h = block.norm1(x) * (1 + scale_msa) + shift_msa
            q = self._rope(block.self_attn.norm_q(block.self_attn.q(h)))
            ka = self._rope(block.self_attn.norm_k(block.self_attn.k(h)))
            va = block.self_attn.v(h)
            x = block.gate(x, gate_msa, block.self_attn.o(self._attention(q, torch.cat((kv[2*i], ka), 1), torch.cat((kv[2*i+1], va), 1), action_mask)))
            cross = block.cross_attn
            h = block.norm3(x)
            x = x + cross.o(self._attention(cross.norm_q(cross.q(h)), cross.norm_k(cross.k(context)), cross.v(context), context_mask.unsqueeze(1)))
            h = block.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = block.gate(x, gate_mlp, block.ffn(h))
        return self.action_expert.head(x)
