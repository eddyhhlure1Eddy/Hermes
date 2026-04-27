"""Custom training kernels for v1 Spatiotemporal Hermes.

Public API:
    is_available()              -> bool. True if flash-attn is importable.
    fa_self_attn(q, k, v, ...)  -> tensor. Plain self-attention via flash-attn.
    fa_cross_attn(q, k, v, ...) -> tensor. Cross-attention via flash-attn.
    fa_self_attn_bias(q, k, v, bias, ...) -> tensor. Self-attn with additive K×K bias.
    fused_layernorm_residual(x, residual, gamma, beta) -> tensor.
        out = LayerNorm(x + residual). Falls back to torch ops if _C unavailable.

    FastSelfAttention(d_model, n_heads, dropout=0.1)
    FastCrossAttention(d_model, n_heads, dropout=0.1)
    FastTransformerBlock(d_model, n_heads, ff_mult=4, dropout=0.1)

Toggle via env var: HERMES_USE_FAST=1
"""

import os

USE_FAST = os.environ.get("HERMES_USE_FAST", "0") == "1"

from .python.flash_attn_ops import (
    is_available,
    fa_self_attn,
    fa_cross_attn,
    fa_self_attn_bias,
)
from .python.fused_ops import fused_layernorm_residual
from .python.modules import FastSelfAttention, FastCrossAttention
from .python.transformer_block import FastTransformerBlock

__all__ = [
    "USE_FAST",
    "is_available",
    "fa_self_attn",
    "fa_cross_attn",
    "fa_self_attn_bias",
    "fused_layernorm_residual",
    "FastSelfAttention",
    "FastCrossAttention",
    "FastTransformerBlock",
]
