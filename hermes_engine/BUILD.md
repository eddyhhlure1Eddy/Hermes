# Build & Run

## Build (on .201, where 6x RTX PRO 6000 Blackwell + nvcc 13.x live)

```bash
cd ~/桌面/her/hermes_engine          # after rsync from Win
source /mnt/wsl-vllm/LLaMA-Factory-Project/.venv/bin/activate

# nvcc is at /usr/local/cuda-13.1/bin/nvcc but not on PATH by default
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -2          # sanity: must show 13.1

TORCH_CUDA_ARCH_LIST=12.0 pip install -e . -v 2>&1 | tail -20
```

The compile step takes ~2-5 min (one .cu file, sm_120 only). If you also
want sm_90 / sm_100 fallbacks: `TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"`.

## Verify against reference

```bash
python -m hermes_engine.bench.verify \
    --hermes-root /home/eddy/桌面/her/Hermes-main \
    --ckpt /home/eddy/桌面/her/Hermes-main/checkpoints/cnb_l40/ddp_5132s_20260425_205719_best.pt \
    --batch 8 --seq 128
```

Expected: `max|d| < 5e-2` for every head; engine should be 1.5-3x faster
than the bmm-fallback reference at B=8, T=128 on Blackwell.

## What's covered / not covered

| Component                         | Status |
|-----------------------------------|--------|
| input projection + bias           | engine |
| event branch (gated fusion)       | engine |
| positional encoding (sinusoidal)  | engine |
| transformer encoder (4 layers)    | engine, fused MHA single-block |
| LayerNorm (D=256)                 | engine, warp-shuffle reduce    |
| GELU / ReLU + bias                | engine, fused                  |
| GEMM (linear / qkv / out / FFN)   | cuBLAS via torch::matmul       |
| conditional embedding lookup      | engine                         |
| fusion + heads                    | engine                         |
| text branch                       | NOT supported (raise on use)   |
| training / backward               | NOT supported (inference only) |

## Known limits

- T (seq length) must be `<= 256` (single-block flash MHA, K/V live in shared mem).
- `D_h` must be `32` (i.e. D=256, num_heads=8) — the kernel template is fixed.
- bf16 only. fp16 / fp8 not implemented.
