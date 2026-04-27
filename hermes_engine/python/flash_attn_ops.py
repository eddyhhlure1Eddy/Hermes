"""Direct flash-attn 2.8 calls — NO SDPA fallback.

If flash-attn is not importable or dtype/device conditions are not met,
we raise RuntimeError immediately so you know at the FIRST forward call
instead of silently running on a slower path.

For the K×K additive bias used by AnchorTransition we use PyTorch's
torch.nn.attention.flex_attention (JIT-compiled, supports arbitrary score_mod).
flash_attn_func itself only supports ALiBi-style bias, not arbitrary (K, K).

Layout convention (flash-attn):
    (B, S, H, d)  — sequence-first within batch, NOT (B, H, S, d) like SDPA.
"""

import os
import torch
import torch._inductor.exc
import torch.nn.functional as F
from typing import Optional

# HERMES_USE_FAST controls whether we route through flash-attn directly.
# Default = 1 (use FA). Set to 0 to force SDPA fallback for A/B benching.
_USE_FAST = os.environ.get("HERMES_USE_FAST", "1") != "0"

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FA_AVAILABLE = _USE_FAST
except ImportError:
    _flash_attn_func = None
    _FA_AVAILABLE = False

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw
    _flex_attention = torch.compile(_flex_attention_raw, dynamic=False)
    _FLEX_AVAILABLE = True
except ImportError:
    _flex_attention = None
    _FLEX_AVAILABLE = False

_backend_logged = False


def _log_backend(backend: str, dtype: torch.dtype, caller: str):
    global _backend_logged
    if not _backend_logged:
        _backend_logged = True
        print(f"[flash_attn_ops] {caller} backend={backend} dtype={dtype} "
              f"FA_AVAILABLE={_FA_AVAILABLE} FLEX_AVAILABLE={_FLEX_AVAILABLE}",
              flush=True)


def _check_fa_available(caller: str):
    if not _FA_AVAILABLE:
        raise RuntimeError(
            f"{caller}: flash-attn is NOT available. "
            f"Install flash-attn or set HERMES_USE_FAST=0 for SDPA fallback. "
            f"_FA_AVAILABLE={_FA_AVAILABLE} _USE_FAST={_USE_FAST}"
        )


def _check_fa_dtype(t: torch.Tensor, caller: str):
    if not (t.is_cuda and t.dtype in (torch.float16, torch.bfloat16)):
        raise RuntimeError(
            f"{caller}: flash-attn requires CUDA + bf16/fp16, got "
            f"device={t.device} dtype={t.dtype}. "
            f"Set HERMES_USE_FAST=0 for SDPA fallback."
        )


def _check_flex_available(caller: str):
    if not _FLEX_AVAILABLE:
        raise RuntimeError(
            f"{caller}: flex_attention is NOT available (requires PyTorch 2.1+). "
            f"Cannot compute attention with arbitrary K×K bias. "
            f"Set HERMES_USE_FAST=0 for SDPA fallback."
        )


def _check_flex_dtype(t: torch.Tensor, caller: str):
    if not (t.is_cuda and t.dtype in (torch.float16, torch.bfloat16)):
        raise RuntimeError(
            f"{caller}: flex_attention requires CUDA + bf16/fp16, got "
            f"device={t.device} dtype={t.dtype}. "
            f"Set HERMES_USE_FAST=0 for SDPA fallback."
        )


def is_available() -> bool:
    return _FA_AVAILABLE


def fa_self_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    _check_fa_available("fa_self_attn")
    _check_fa_dtype(q, "fa_self_attn")
    _log_backend("flash_attn", q.dtype, "fa_self_attn")
    return _flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def fa_cross_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    _check_fa_available("fa_cross_attn")
    _check_fa_dtype(q, "fa_cross_attn")
    _log_backend("flash_attn", q.dtype, "fa_cross_attn")
    return _flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=False,
    )


# ── Self-attn with K×K additive bias (AnchorTransition) ───────────────

def _make_bias_score_mod(bias_kk: torch.Tensor):
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_kk[q_idx, kv_idx]
    return score_mod


def _manual_attn_with_bias(q, k, v, bias_kk, dropout_p, scale):
    """Pure-torch QK^T softmax matmul, with K×K additive bias.

    NOT SDPA. NOT flex. Just fp16/bf16 matmul + softmax + matmul. Used as
    a fallback for fa_self_attn_bias when flex_attention's triton backward
    can't fit in shared memory (Blackwell sm_120 P3 case).

    Input layout: flash-attn convention (B, S, H, d).
    Internally transposes to (B, H, S, d) for bmm, then transposes back.
    Backward via autograd over standard ops.
    """
    # (B, S, H, d) -> (B, H, S, d) for bmm
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale          # (B,H,K,K)
    scores = scores + bias_kk.unsqueeze(0).unsqueeze(0).to(scores.dtype)
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    out = torch.matmul(attn, v)                                   # (B,H,K,d)
    return out.transpose(1, 2)                                    # (B,K,H,d)


def fa_self_attn_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias_kk: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Self-attention with arbitrary K×K additive bias (AnchorTransition).

    flash_attn doesn't accept arbitrary bias (only ALiBi).
    flex_attention is the preferred path (compiled triton kernel) but on
    Blackwell sm_120 with d_head=64 + n_heads=12 + K=48 (P3) the BACKWARD
    triton kernel can't fit in shared memory: Required 114688 / Hardware
    limit 101376 → "No valid triton configs". When flex compile fails we
    drop to manual fp16/bf16 matmul + softmax + matmul (autograd handles
    backward). Still NO SDPA path.

    K safety check: K>128 would allocate (B,H,K,K) which is multiple GB
    per layer. Fail early with a clear error rather than OOM mid-training.
    """
    K = q.size(2)
    if K > 128:
        raise RuntimeError(
            f"fa_self_attn_bias: K={K} exceeds safety limit of 128. "
            f"K×K attention bias allocates (B,H,{K},{K}) per layer — "
            f"that's {K*K*4/1e6:.1f}M elements per head. "
            f"Reduce n_anchors to <=128 or use a different attention path."
        )
    scale = softmax_scale if softmax_scale is not None else (1.0 / (q.size(-1) ** 0.5))
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
        raise RuntimeError(
            f"fa_self_attn_bias: requires CUDA + bf16/fp16, got "
            f"device={q.device} dtype={q.dtype}"
        )
    # NOTE: flex_attention was the original choice but its compiled triton
    # backward kernel allocates more shared memory than sm_120 (Blackwell)
    # provides for d_head=64 + n_heads=12 + K=48 (P3 config). The compile
    # failure triggers during backward, AFTER the flex_attention call has
    # returned, so a try/except around the call doesn't catch it. Always
    # use manual matmul on this scale (K<=64 is tiny anyway, ~1.8M flops).
    _log_backend("manual matmul (no SDPA, no flex)", q.dtype, "fa_self_attn_bias")
    return _manual_attn_with_bias(q, k, v, bias_kk, dropout_p, scale)
