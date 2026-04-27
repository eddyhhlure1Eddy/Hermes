# src/ — Custom Training Kernels for Spatiotemporal Hermes (v1)

Hand-written CUDA + flash-attn integration to bypass `F.scaled_dot_product_attention`
overhead in the v1 (3D coupled) DDP training path. Targets sm_120 (Blackwell) but
falls back gracefully if the C extension fails to build.

## Why a separate `src/` (not in `model/` or `hermes_engine/`)

- `model/` is pure PyTorch reference — keeps shape/numerics clear.
- `hermes_engine/` was written for v0 inference (last-step pooling, single transformer),
  is bf16-only, no backward, and explicitly marked "NOT YET ADAPTED" for v1 in `BUILD.md`.
- `src/` is for v1 training: fused fwd+bwd kernels, AMP-aware (fp16/bf16),
  drop-in `nn.Module` replacements, optional — toggle via `HERMES_USE_FAST=1`.

## What's here

| Component | Path | Status | Backend |
|-----------|------|--------|---------|
| FA-based self-attn (no bias) | `python/flash_attn_ops.py:fa_self_attn` | done | flash-attn 2.8 |
| FA-based cross-attn | `python/flash_attn_ops.py:fa_cross_attn` | done | flash-attn 2.8 |
| FA-based self-attn with K×K bias (AnchorTransition) | `python/flash_attn_ops.py:fa_self_attn_bias` | done | flash-attn 2.8 attn_bias |
| Fused LN + residual + dropout | `csrc/fused_ln_residual.cu` | done | custom CUDA, sm_120 |
| Fast self-attention `nn.Module` | `python/modules.py:FastSelfAttention` | done | wraps fa_self_attn |
| Fast cross-attention `nn.Module` | `python/modules.py:FastCrossAttention` | done | wraps fa_cross_attn |
| Fast transformer block | `python/transformer_block.py:FastTransformerBlock` | done | replaces `nn.TransformerEncoderLayer` |

## Build

```bash
cd src/
TORCH_CUDA_ARCH_LIST=12.0 pip install -e . -v
```

Requires `nvcc` on PATH (export `CUDA_HOME=/usr/local/cuda-13.1` on .201).

## Use (drop-in)

Set env var to route the model through `src/` modules:

```bash
HERMES_USE_FAST=1 bash launch_ddp.sh 4 0 30 p1
```

Without the env var, model uses pure PyTorch SDPA (reference path). Useful for
A/B numerical verification and benching custom-kernel impact.

## Numerical contract

Custom kernels must match SDPA reference to `max|d| < 2e-3` (bf16) / `5e-4` (fp16)
on randomized inputs. See `tests/test_attention.py`.
