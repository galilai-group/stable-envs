"""cProfile of a real render at the operating point, so the remainder is ATTRIBUTED not guessed.

Every pass-level probe so far measures the passes and leaves a remainder, and the remainder is now
most of the render: the fused casts and the bounded windows are ~4 ms of a 25 ms chunk and the
composite tail micro-benchmarks at 0.4 ms. cProfile names the rest, including the numpy and
host-to-device work that a CUDA-synchronised timer around a pass cannot see.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
import torch

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="DMCCartpoleBalance,DMCBallInCupCatch,DMCFingerTurnEasy")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--rows", type=int, default=22)
a = ap.parse_args()

for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    env = se.make(name, batch=a.batch, size=a.size, device="cuda")
    s = env.sample_states(a.batch, rng=np.random.default_rng(0))
    for _ in range(2):
        env.render(s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    env.render(s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"\n{'=' * 90}\n{name}  {a.size}px b{a.batch}: {dt * 1e3:.1f} ms = {a.batch / dt:,.0f} f/s")

    pr = cProfile.Profile()
    pr.enable()
    env.render(s)
    torch.cuda.synchronize()
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(a.rows)
    for line in out.getvalue().splitlines():
        t = line.strip()
        if t and "function calls" not in t and not t.startswith(("Ordered by", "Random listing")):
            print("   " + line.rstrip())
    del env
    torch.cuda.empty_cache()
