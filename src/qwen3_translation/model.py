"""Hand-written Qwen3 Dense components extracted unchanged from the completed experiment.

See NOTICE.md for the assignment scaffold and architectural references.
"""


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QwenDenseConfig:
    vocab_size: int
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 32
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.num_key_value_heads < self.num_attention_heads
        assert self.head_dim % 2 == 0
        assert self.vocab_size > 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Float32 reduction prevents overflow/underflow in mixed precision.
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=1_000_000.0):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, position_ids):
        # [B,T,dh/2] -> duplicated split-half angles [B,1,T,dh].
        with torch.autocast(device_type=q.device.type, enabled=False):
            half_angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((half_angles, half_angles), dim=-1).unsqueeze(1)
            cos, sin = angles.cos(), angles.sin()

        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            rotated_half = torch.cat((-second, first), dim=-1)
            return x * cos.to(x.dtype) + rotated_half * sin.to(x.dtype)

        return rotate(q), rotate(k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden_states, attention_mask, position_ids):
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # One learned dh-wide scale per Q/K norm, shared across its heads.
        q, k = self.rope(self.q_norm(q), self.k_norm(k), position_ids)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(length, device=hidden_states.device)
        causal = positions[None, :] <= positions[:, None]  # key s <= query t
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.o_proj(attended.view(batch, length, self.num_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDenseDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, attention_mask, position_ids):
        x = x + self.attention(self.attention_norm(x), attention_mask, position_ids)
        return x + self.mlp(self.mlp_norm(x))


class QwenDenseForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QwenDenseDecoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        batch, length = input_ids.shape
        if not 0 < length <= self.config.max_position_embeddings:
            raise ValueError("Sequence length must be positive and fit the context window")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        # Right padding and nonempty prompts guarantee at least one valid key.
        position_ids = torch.arange(length, device=input_ids.device).expand(batch, -1)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        return self.lm_head(self.final_norm(x))
