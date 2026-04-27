"""FastTransformerBlock — replacement for nn.TransformerEncoderLayer.

PyTorch's nn.TransformerEncoderLayer / nn.TransformerEncoder is generic
(supports masking, batched/unbatched, nested tensors, etc.) and the dispatch
overhead per layer is non-trivial. We hand-roll the same math (pre-norm,
self-attn, residual, pre-norm, FFN, residual) using flash-attn directly +
the fused LN-residual kernel when available.

When CUDA + bf16/fp16 + hermes_fast._C is available, we switch to POST-norm:
    out = LN(x + drop(SelfAttn(x)))
    out = LN(out + FFN(out))
This matches the fused_ln_residual kernel math exactly.

When the fused kernel is unavailable, we fall back to standard PRE-norm:
    out = x + drop(SelfAttn(LN1(x)))
    out = out + FFN(LN2(out))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FastSelfAttention
from .fused_ops import fused_layernorm_residual, native_available


class _FastFFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class FastTransformerBlock(nn.Module):
    """Transformer block that uses fused LN+residual (post-norm) when
    hermes_fast._C is available, otherwise falls back to pre-norm PyTorch ops.

    Forward: x: (B, S, D) → (B, S, D)

    Supports gradient checkpointing to reduce peak VRAM.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.ln_eps = ln_eps
        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.attn = FastSelfAttention(d_model, n_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.ff = _FastFFN(d_model, mult=ff_mult, dropout=dropout)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        use_fused = (x.is_cuda
                     and x.dtype in (torch.float16, torch.bfloat16)
                     and native_available())
        D = x.size(-1)

        if use_fused:
            # Post-norm: out = LN(x + drop(Attn(x)))
            attn_out = self.attn_drop(self.attn(x))
            x = fused_layernorm_residual(
                attn_out, x,
                self.ln1.weight, self.ln1.bias, self.ln_eps,
            )
            # Post-norm: out = LN(x + FFN(x))
            ff_out = self.ff(x)
            x = fused_layernorm_residual(
                ff_out, x,
                self.ln2.weight, self.ln2.bias, self.ln_eps,
            )
        else:
            # Pre-norm fallback
            attn_out = self.attn_drop(self.attn(self.ln1(x)))
            x = x + attn_out
            ff_out = self.ff(self.ln2(x))
            x = x + ff_out
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class FastTransformerEncoder(nn.Module):
    """Stack of FastTransformerBlock — replaces nn.TransformerEncoder.

    The loss of `nn.TransformerEncoder.enable_nested_tensor` doesn't matter
    here because our inputs are dense (no padding masks).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FastTransformerBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
