"""Numerical match: src/ attention modules vs PyTorch SDPA reference.

Run on .201:
    cd /home/eddy/桌面/hermes_st
    source venv/bin/activate
    PYTHONPATH=. python -m src.tests.test_attention
"""

import math
import torch
import torch.nn.functional as F


def _ref_self_attn(q, k, v, scale=None):
    """Reference self-attention via SDPA. q,k,v: (B, S, H, d)."""
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    s = scale if scale is not None else (1.0 / (q.size(-1) ** 0.5))
    out = F.scaled_dot_product_attention(qh, kh, vh, scale=s)
    return out.transpose(1, 2).contiguous()


def test_fa_self_attn():
    from src.python.flash_attn_ops import fa_self_attn, is_available
    assert is_available(), "flash-attn not installed"
    torch.manual_seed(0)
    B, S, H, d = 4, 60, 8, 32
    q = torch.randn(B, S, H, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, d, device="cuda", dtype=torch.bfloat16)
    out_fa = fa_self_attn(q, k, v)
    out_ref = _ref_self_attn(q, k, v)
    err = (out_fa.float() - out_ref.float()).abs().max().item()
    print(f"[fa_self_attn] max|d| = {err:.2e}")
    assert err < 1e-2, f"diff too large: {err}"


def test_fa_cross_attn():
    from src.python.flash_attn_ops import fa_cross_attn
    torch.manual_seed(1)
    B, Sq, Skv, H, d = 4, 64, 60, 8, 32
    q = torch.randn(B, Sq, H, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, d, device="cuda", dtype=torch.bfloat16)
    out_fa = fa_cross_attn(q, k, v)
    # reference: SDPA with same inputs
    qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
    out_ref = F.scaled_dot_product_attention(qh, kh, vh, scale=1.0/(d**0.5)).transpose(1, 2).contiguous()
    err = (out_fa.float() - out_ref.float()).abs().max().item()
    print(f"[fa_cross_attn] max|d| = {err:.2e}")
    assert err < 1e-2, f"diff too large: {err}"


def test_fa_self_attn_bias():
    from src.python.flash_attn_ops import fa_self_attn_bias
    torch.manual_seed(2)
    B, H, K, d = 4, 4, 64, 64
    q = torch.randn(B, H, K, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, K, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, K, d, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(K, K, device="cuda", dtype=torch.bfloat16) * 0.1
    out = fa_self_attn_bias(q, k, v, bias_kk=bias)
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=bias.unsqueeze(0).unsqueeze(0).to(q.dtype),
        scale=1.0/(d**0.5),
    )
    err = (out.float() - out_ref.float()).abs().max().item()
    print(f"[fa_self_attn_bias] max|d| = {err:.2e}")
    assert err < 1e-2, f"diff too large: {err}"


def test_fast_self_attention_module():
    from src.python.modules import FastSelfAttention
    torch.manual_seed(3)
    B, S, D, H = 4, 60, 256, 8
    m = FastSelfAttention(D, H, dropout=0.0).cuda().to(torch.bfloat16)
    m.eval()
    x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape
    print(f"[FastSelfAttention] shape OK {out.shape}")


def test_fused_ln_residual():
    from src.python.fused_ops import fused_layernorm_residual, native_available
    torch.manual_seed(4)
    if not native_available():
        print("[fused_ln_residual] native not built — testing fallback only")
    B, S, D = 4, 60, 256
    x   = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    g   = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    b   = torch.zeros(D, device="cuda", dtype=torch.bfloat16)
    out = fused_layernorm_residual(x, res, g, b, eps=1e-5)
    out_ref = F.layer_norm((x + res), (D,), g, b, eps=1e-5)
    err = (out.float() - out_ref.float()).abs().max().item()
    print(f"[fused_ln_residual] max|d| = {err:.2e}, native={native_available()}")
    # bf16 precision: at output scale ~3 (3-sigma), bf16 LSB ≈ 0.023.
    # Half-LSB roundoff is the best achievable. 3e-2 = ~1 LSB tolerance.
    assert err < 3e-2, f"diff too large: {err}"


if __name__ == "__main__":
    torch.cuda.set_device(0)
    test_fa_self_attn()
    test_fa_cross_attn()
    test_fa_self_attn_bias()
    test_fast_self_attention_module()
    test_fused_ln_residual()
    print("\nALL TESTS PASSED")
