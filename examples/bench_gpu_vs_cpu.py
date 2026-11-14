"""Time ``env.render(state)`` for the same states on CPU and on CUDA.

Each env is reset with ``--pool`` lanes; the state is tiled up to each batch size, since ``render``
accepts a state of any batch. The table reports frames per second per device and the CUDA speedup.
Envs that require CUDA have no CPU column. Without CUDA only the CPU column is measured.

    python examples/bench_gpu_vs_cpu.py
    python examples/bench_gpu_vs_cpu.py --envs CartPole,PushT --batches 1,64,1024 --out results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as snv


def seconds(fn, device: str, reps: int) -> float:
    """Median wall seconds of ``fn`` after one warm-up call, synchronized on CUDA."""
    fn()
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def measure(name: str, device: str, batches: list[int], pool: int, reps: int) -> dict[int, float]:
    """Frames per second of ``render`` at each batch size."""
    fps = {}
    with snv.make(name, num_envs=pool, device=device) as env:
        state = env.reset(seed=0)
        for b in batches:
            batch = state[torch.arange(b) % pool]
            fps[b] = b / seconds(lambda batch=batch: env.render(batch), device, reps)
    return fps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="render() frames/s on CPU and CUDA for the same States.")
    p.add_argument("--envs", default="", help="comma-separated names; default is every registered env")
    p.add_argument("--batches", default="1,64,1024")
    p.add_argument("--pool", type=int, default=16, help="lanes reset per env; States are tiled to each batch")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out", type=Path, default=None, help="optional JSON path for the results")
    args = p.parse_args(argv)

    chosen = [s for s in args.envs.split(",") if s]
    names = chosen or snv.list_envs()
    batches = [int(b) for b in args.batches.split(",") if b]
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    results = {}
    print("| env | batch | " + " | ".join(f"{d} frames/s" for d in devices) + " | cuda/cpu |")
    print("|---|---|" + "---|" * len(devices) + "---|")
    for name in names:
        cuda_only = snv.info(name)["requires_cuda"]
        try:
            per_device = {
                d: measure(name, d, batches, args.pool, args.reps) for d in devices if d == "cuda" or not cuda_only
            }
        except (ImportError, RuntimeError) as e:
            # Without --envs, skip the classes whose extra is missing on this machine.
            if chosen:
                raise
            print(f"skipped {name}: {e}")
            continue
        results[name] = per_device
        for b in batches:
            cells = [f"{per_device[d][b]:,.0f}" if d in per_device else "-" for d in devices]
            ratio = per_device["cuda"][b] / per_device["cpu"][b] if len(per_device) == 2 else None
            print(f"| {name} | {b} | " + " | ".join(cells) + f" | {'-' if ratio is None else f'{ratio:.1f}x'} |")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
