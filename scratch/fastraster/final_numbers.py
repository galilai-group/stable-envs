"""End-to-end throughput at each env's native size and at 256px, for the report."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se

HIGHWAY = ["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"]


def fps(env, st, B, size=None, reps=9):
    for _ in range(3):
        env.render(st, size=size)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(st, size=size)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    return B / t, 1000 * t


def main(B=1024):
    print(f"{'env':14s} {'size':>6s} {'frame':>12s} {'ms':>9s} {'f/s':>11s}", flush=True)
    for name in HIGHWAY:
        env = se.make(name, batch=1, device="cuda")
        st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)), device="cuda")
        for size in (None, 256):
            scr = env.screen(size)
            f, ms = fps(env, st, B, size)
            tag = "native" if size is None else "256"
            print(f"{name:14s} {tag:>6s} {str((scr.H, scr.W)):>12s} {ms:9.2f} {f:11.1f}", flush=True)
        del env
        torch.cuda.empty_cache()
    try:
        env = se.make("CarRacing", batch=1, device="cuda")
        st = torch.as_tensor(env.sample_states(B), device="cuda")
        f, ms = fps(env, st, B, None, reps=5)
        print(f"{'CarRacing':14s} {'native':>6s} {'(96, 96)':>12s} {ms:9.2f} {f:11.1f}", flush=True)
    except Exception as e:
        print(f"CarRacing: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    main()
