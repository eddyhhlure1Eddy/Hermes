"""
Numerical verification: HermesEngine vs PyTorch reference (ConditionalFinancialModel).

Usage:
    python -m hermes_engine.bench.verify \
        --hermes-root /home/eddy/桌面/her/Hermes-main \
        --ckpt checkpoints/cnb_l40/ddp_5132s_xxx_best.pt \
        --batch 8 --seq 128

Compares each output head; prints max-abs-diff and cosine similarity.
A correct engine should reach < 5e-2 max-abs-diff in bf16
(LayerNorm + softmax in fp32 accumulator gives ~1-3 ULP per op, accumulated
over 4 layers is the dominant source).
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import torch

import hermes_engine as he


def _add_hermes_to_sys_path(hermes_root: str) -> None:
    fm = os.path.join(hermes_root, "financial_model")
    for p in (hermes_root, fm):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_reference_model(state_dict, device):
    from src.conditional_model import ConditionalFinancialModel

    has_event = "event_projection.0.weight" in state_dict
    num_events = 0
    if has_event:
        num_events = state_dict["event_projection.0.weight"].shape[1]

    D = state_dict["input_projection.weight"].shape[0]
    in_dim = state_dict["input_projection.weight"].shape[1]
    n_ind = state_dict["industry_embedding.weight"].shape[0]
    n_sty = state_dict["style_embedding.weight"].shape[0]
    n_reg = state_dict["regime_embedding.weight"].shape[0]
    n_cg  = state_dict["cg_state_embedding.weight"].shape[0]
    n_rr  = state_dict["risk_regime_embedding.weight"].shape[0]

    model = ConditionalFinancialModel(
        input_dim=in_dim,
        hidden_dim=D,
        num_layers=4,
        num_heads=8,
        num_industries=n_ind,
        num_style_factors=n_sty,
        num_market_regimes=n_reg,
        num_cg_states=n_cg,
        num_risk_regimes=n_rr,
        dropout=0.0,
        pred_length=state_dict["prediction_head.3.weight"].shape[0],
        use_flash_attention=False,            # use bmm fallback for stable comparison
        industry_embed_dim=state_dict["industry_embedding.weight"].shape[1],
        style_embed_dim=state_dict["style_embedding.weight"].shape[1],
        regime_embed_dim=state_dict["regime_embedding.weight"].shape[1],
        cg_state_embed_dim=state_dict["cg_state_embedding.weight"].shape[1],
        risk_regime_embed_dim=state_dict["risk_regime_embedding.weight"].shape[1],
        use_multi_task=True,
        num_event_categories=num_events,
        event_gate_bias=-2.0,
        text_embed_dim=0,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ref model: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).to(torch.bfloat16).eval(), num_events


def _diff_summary(a: torch.Tensor, b: torch.Tensor) -> str:
    a = a.float(); b = b.float()
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return f"max|d|={diff.max().item():.4e}  mean|d|={diff.mean().item():.4e}  cos={cos:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hermes-root", required=True,
                    help="Hermes-main project root (so we can import src.conditional_model)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq",   type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=50)
    args = ap.parse_args()

    _add_hermes_to_sys_path(args.hermes_root)

    device = torch.device(args.device)
    torch.manual_seed(42)

    print(f"loading checkpoint: {args.ckpt}")
    sd_obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd_obj
    if isinstance(sd_obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd_obj and isinstance(sd_obj[k], dict):
                sd = sd_obj[k]
                break
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    print("building reference model")
    ref, num_events = _build_reference_model(sd, device)

    print("building Hermes engine")
    eng = he.HermesEngine.from_state_dict(sd, device=device, num_layers=4)
    print(f"  has_event_branch={eng.has_event_branch}  params={eng.num_parameters()/1e6:.2f}M")

    # ---------------------------------------------------------------- inputs
    B, T = args.batch, args.seq
    in_dim = sd["input_projection.weight"].shape[1]
    n_ind = sd["industry_embedding.weight"].shape[0]
    n_sty = sd["style_embedding.weight"].shape[0]
    n_reg = sd["regime_embedding.weight"].shape[0]
    n_cg  = sd["cg_state_embedding.weight"].shape[0]
    n_rr  = sd["risk_regime_embedding.weight"].shape[0]

    x = torch.randn(B, T, in_dim, dtype=torch.bfloat16, device=device)
    ind = torch.randint(0, n_ind, (B,), dtype=torch.int64, device=device)
    sty = torch.randint(0, n_sty, (B,), dtype=torch.int64, device=device)
    reg = torch.randint(0, n_reg, (B,), dtype=torch.int64, device=device)
    cgs = torch.randint(0, n_cg,  (B,), dtype=torch.int64, device=device)
    rsk = torch.randint(0, n_rr,  (B,), dtype=torch.int64, device=device)
    events = None
    if num_events > 0:
        events = torch.randn(B, T, num_events, dtype=torch.bfloat16, device=device)

    # ---------------------------------------------------------------- forward
    print("running reference forward")
    with torch.inference_mode():
        ref_out = ref(x, ind, sty, reg, cg_state_idx=cgs, risk_regime_idx=rsk,
                      event_scores=events)

    print("running engine forward")
    eng_out = eng.forward(x, ind, sty, reg, cgs, rsk, events=events)

    # ---------------------------------------------------------------- diff
    print("\n=== numerical diff (engine vs reference) ===")
    for k in ("price", "cg_state", "cg_score", "risk_regime", "signal", "direction"):
        if k not in ref_out:
            continue
        print(f"  {k:<12}  {_diff_summary(eng_out[k], ref_out[k])}")

    # ---------------------------------------------------------------- speed
    torch.cuda.synchronize()
    print(f"\n=== latency benchmark (B={B}, T={T}) ===")

    for _ in range(args.warmup):
        _ = eng.forward(x, ind, sty, reg, cgs, rsk, events=events)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        _ = eng.forward(x, ind, sty, reg, cgs, rsk, events=events)
    torch.cuda.synchronize()
    dt_eng = (time.time() - t0) / args.iters * 1000

    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = ref(x, ind, sty, reg, cg_state_idx=cgs, risk_regime_idx=rsk,
                    event_scores=events)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        with torch.inference_mode():
            _ = ref(x, ind, sty, reg, cg_state_idx=cgs, risk_regime_idx=rsk,
                    event_scores=events)
    torch.cuda.synchronize()
    dt_ref = (time.time() - t0) / args.iters * 1000

    print(f"  reference (bf16, bmm attn): {dt_ref:6.2f} ms / iter")
    print(f"  hermes engine             : {dt_eng:6.2f} ms / iter   ({dt_ref/dt_eng:.2f}x)")


if __name__ == "__main__":
    main()
