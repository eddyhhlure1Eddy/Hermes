"""Drop-in replacements for the SDPA-based attention modules in model/.

Each Module here matches the public API of its model/ counterpart so that
the model can swap them in via `from src import FastSelfAttention as ...`.

Why these exist:
  - Bypass F.scaled_dot_product_attention dispatch overhead.
  - Use flash-attn 2.8 directly with the (B, S, H, d) layout.
  - Fuse pre-norm + residual via the fused_layernorm_residual op when the
    extension is built.

These modules deliberately do NOT include the residual+norm dance — that
stays in the calling block (so we don't accidentally double-add residual).
"""

import torch
import torch.nn as nn
from .flash_attn_ops import fa_self_attn, fa_cross_attn


class FastSelfAttention(nn.Module):
    """Self-attention via flash-attn. Matches the math of the SDPA path in
    model/spatiotemporal_block.py:TemporalAttention / SpatialAttentionWithBias.

    Forward expects x in (B, S, D) layout. Returns (B, S, D).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, d = self.n_heads, self.head_dim
        # flash-attn layout: (B, S, H, d) — no transpose to (B, H, S, d) needed.
        q = self.q_proj(x).view(B, S, H, d)
        k = self.k_proj(x).view(B, S, H, d)
        v = self.v_proj(x).view(B, S, H, d)
        out = fa_self_attn(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, S, H, d)
        out = out.reshape(B, S, D)
        return self.out_proj(out)


class FastCrossAttention(nn.Module):
    """Cross-attention via flash-attn. Q has length Sq, KV has length Skv.

    Used by EventCrossAttention (Q=grid_flat T*K, KV=event T_ev) and
    _AggLayer (Q=anchors K, KV=temporal_or_event T).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Sq, D = q.shape
        Skv = kv.size(1)
        H, d = self.n_heads, self.head_dim
        qh = self.q_proj(q).view(B, Sq, H, d)
        kh = self.k_proj(kv).view(B, Skv, H, d)
        vh = self.v_proj(kv).view(B, Skv, H, d)
        out = fa_cross_attn(
            qh, kh, vh,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.reshape(B, Sq, D)
        return self.out_proj(out)
