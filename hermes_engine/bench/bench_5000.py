"""
End-to-end 5000-stock x 60-day inference benchmark using HermesEngine.

Single GPU mode (default):
    python bench/bench_5000.py --ckpt <path> --batch 512

Multi-GPU mode (split codes across N GPUs):
    python bench/bench_5000.py --ckpt <path> --batch 512 --gpus 6

Reports wall-clock time for the inference part only (no parquet IO,
no pandas preprocessing). 5000 stocks * 60 days = 300,000 forward calls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import torch.multiprocessing as mp


def _bench_single_gpu(ckpt: str, gpu_id: int, n_stocks: int, n_days: int,
                      seq: int, batch: int, num_layers: int = 4,
                      hermes_root: str = "") -> dict:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    if hermes_root:
        for p in (hermes_root, os.path.join(hermes_root, "financial_model")):
            if p not in sys.path:
                sys.path.insert(0, p)

    import hermes_engine as he

    eng = he.HermesEngine.from_checkpoint(ckpt, device=device, num_layers=num_layers)

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

    in_dim = sd["input_projection.weight"].shape[1]
    n_ind = sd["industry_embedding.weight"].shape[0]
    n_sty = sd["style_embedding.weight"].shape[0]
    n_reg = sd["regime_embedding.weight"].shape[0]
    n_cg = sd["cg_state_embedding.weight"].shape[0]
    n_rr = sd["risk_regime_embedding.weight"].shape[0]
    has_events = "event_projection.0.weight" in sd
    n_ev = sd["event_projection.0.weight"].shape[1] if has_events else 0
    pred_len = sd["prediction_head.3.weight"].shape[0]

    print(f"[GPU {gpu_id}] params={eng.num_parameters()/1e6:.2f}M  "
          f"in_dim={in_dim} pred_len={pred_len} events={n_ev}")

    total = n_stocks * n_days
    n_batches = (total + batch - 1) // batch

    # Pre-allocate one set of inputs reused across batches (we measure
    # inference cost, not host-device transfer cost). Industry/style
    # vary per stock so we round-robin across stocks for each batch.
    x_buf      = torch.empty(batch, seq, in_dim, dtype=torch.bfloat16, device=device)
    ind_buf    = torch.empty(batch, dtype=torch.int64, device=device)
    sty_buf    = torch.empty(batch, dtype=torch.int64, device=device)
    reg_buf    = torch.empty(batch, dtype=torch.int64, device=device)
    cgs_buf    = torch.empty(batch, dtype=torch.int64, device=device)
    rsk_buf    = torch.empty(batch, dtype=torch.int64, device=device)
    ev_buf     = torch.empty(batch, seq, n_ev, dtype=torch.bfloat16, device=device) if has_events else None

    # initialise random data once on device
    torch.manual_seed(42 + gpu_id)
    x_buf.normal_()
    ind_buf.random_(0, n_ind)
    sty_buf.random_(0, n_sty)
    reg_buf.random_(0, n_reg)
    cgs_buf.random_(0, n_cg)
    rsk_buf.random_(0, n_rr)
    if ev_buf is not None:
        ev_buf.normal_()

    # warmup
    for _ in range(3):
        _ = eng.forward(x_buf, ind_buf, sty_buf, reg_buf, cgs_buf, rsk_buf, events=ev_buf)
    torch.cuda.synchronize(device)

    # main loop
    t0 = time.time()
    samples_done = 0
    for bi in range(n_batches):
        cur = batch if (bi + 1) * batch <= total else total - bi * batch
        if cur != batch:
            x_in   = x_buf[:cur]
            ind_in = ind_buf[:cur]
            sty_in = sty_buf[:cur]
            reg_in = reg_buf[:cur]
            cgs_in = cgs_buf[:cur]
            rsk_in = rsk_buf[:cur]
            ev_in  = ev_buf[:cur] if ev_buf is not None else None
        else:
            x_in, ind_in, sty_in, reg_in, cgs_in, rsk_in, ev_in = \
                x_buf, ind_buf, sty_buf, reg_buf, cgs_buf, rsk_buf, ev_buf
        _ = eng.forward(x_in, ind_in, sty_in, reg_in, cgs_in, rsk_in, events=ev_in)
        samples_done += cur
    torch.cuda.synchronize(device)
    dt = time.time() - t0

    return {
        "gpu": gpu_id,
        "n_batches": n_batches,
        "samples": samples_done,
        "wall_s": dt,
        "throughput": samples_done / dt,
        "ms_per_batch": dt / n_batches * 1000,
    }


def _gpu_worker(gpu_id, ckpt, n_stocks, n_days, seq, batch, num_layers, hermes_root, q):
    res = _bench_single_gpu(ckpt, gpu_id, n_stocks, n_days, seq, batch, num_layers, hermes_root)
    q.put(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hermes-root", default="/home/eddy/桌面/her/Hermes-main",
                    help="Hermes-main path so ckpt can find src.* classes during torch.load")
    ap.add_argument("--n-stocks", type=int, default=5000)
    ap.add_argument("--n-days",   type=int, default=60)
    ap.add_argument("--seq",      type=int, default=128)
    ap.add_argument("--batch",    type=int, default=512)
    ap.add_argument("--gpus",     type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=4)
    args = ap.parse_args()

    total = args.n_stocks * args.n_days
    print(f"Workload: {args.n_stocks} stocks x {args.n_days} days = {total:,} forward calls")
    print(f"Config:   seq={args.seq}, batch={args.batch}, gpus={args.gpus}, num_layers={args.num_layers}")
    print()

    if args.gpus == 1:
        res = _bench_single_gpu(args.ckpt, 0, args.n_stocks, args.n_days,
                                args.seq, args.batch, args.num_layers,
                                args.hermes_root)
        print()
        print(f"=== single GPU ===")
        print(f"  wall:        {res['wall_s']:.2f} s")
        print(f"  throughput:  {res['throughput']:,.0f} samples/s")
        print(f"  per batch:   {res['ms_per_batch']:.2f} ms")
        print(f"  pass <3min:  {'YES' if res['wall_s'] < 180 else 'NO'}")
        return

    # multi-GPU: split codes evenly
    per_gpu_stocks = (args.n_stocks + args.gpus - 1) // args.gpus
    print(f"Per GPU: {per_gpu_stocks} stocks x {args.n_days} days = {per_gpu_stocks*args.n_days:,} forwards")
    print()

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    procs = []
    for g in range(args.gpus):
        p = mp.Process(target=_gpu_worker,
                       args=(g, args.ckpt, per_gpu_stocks, args.n_days,
                             args.seq, args.batch, args.num_layers,
                             args.hermes_root, q))
        p.start()
        procs.append(p)

    t0 = time.time()
    results = []
    for _ in procs:
        results.append(q.get())
    for p in procs:
        p.join()
    wall = time.time() - t0

    print()
    print(f"=== {args.gpus} GPUs ===")
    for r in sorted(results, key=lambda x: x["gpu"]):
        print(f"  GPU {r['gpu']}: {r['samples']:,} in {r['wall_s']:.2f}s "
              f"({r['throughput']:,.0f} samples/s)")
    total_samples = sum(r["samples"] for r in results)
    print(f"  total wall:      {wall:.2f} s")
    print(f"  agg throughput:  {total_samples/wall:,.0f} samples/s")
    print(f"  pass <3min:      {'YES' if wall < 180 else 'NO'}")


if __name__ == "__main__":
    main()
