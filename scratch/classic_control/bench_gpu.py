"""Throughput: batched GPU rasterizer vs the official pygame renderer, same frames.

Reported in frames/s and in Gpixel/s, because classic-control frames are 25x-60x larger than the
96x96 PushT frames the 14.1M frames/s number was measured on -- frames/s alone is not comparable
across resolutions, and quoting it as if it were would flatter this by an order of magnitude.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym

import gpu_render as R

CASES = [("CartPole-v1", R.CartPoleGPU, lambda n, g: np.stack(
    [g.uniform(-2.0, 2.0, n), g.uniform(-0.2, 0.2, n)], 1)),
    ("Pendulum-v1", R.PendulumGPU, lambda n, g: g.uniform(-np.pi, np.pi, (n, 1))),
    ("Acrobot-v1", R.AcrobotGPU, lambda n, g: g.uniform(-np.pi, np.pi, (n, 2))),
    ("MountainCar-v0", R.MountainCarGPU, lambda n, g: g.uniform(-1.2, 0.6, (n, 1)))]


def cpu_fps(name, n=200):
    env = gym.make(name, render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    for _ in range(10):
        env.render()
    t0 = time.perf_counter()
    for _ in range(n):
        env.render()
    dt = time.perf_counter() - t0
    env.close()
    return n / dt


def gpu_fps(cls, B, aa, reps, device, kw=None):
    rend = cls(device=device, aa=aa, **(kw or {}))
    g = np.random.default_rng(0)
    st = CASES_BY_CLS[cls](B, g)
    st_t = torch.as_tensor(st, dtype=torch.float64, device=device)
    for _ in range(3):
        rend.render(st_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = rend.render(st_t)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    px = cls.H * cls.W
    return B * reps / dt, B * reps * px / dt / 1e9, out.shape


CASES_BY_CLS = {c: f for _, c, f in CASES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024, 4096])
    args = ap.parse_args()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"{'env':16s} {'HxW':10s} {'mode':6s} {'B':>6s} {'frames/s':>12s} {'Gpix/s':>9s} "
          f"{'vs pygame':>10s}")
    print("-" * 78)
    for name, cls, _f in CASES:
        c = cpu_fps(name)
        print(f"{name:16s} {f'{cls.H}x{cls.W}':10s} {'pygame':6s} {1:6d} {c:12.1f} "
              f"{c*cls.H*cls.W/1e9:9.4f} {'1.0x':>10s}")
        for aa in (True, False):
            for B in args.batches:
                try:
                    f, gp, shp = gpu_fps(cls, B, aa, args.reps, dev)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{'':16s} {'':10s} {'aa' if aa else 'noaa':6s} {B:6d}  OOM")
                    continue
                print(f"{'':16s} {'':10s} {'aa' if aa else 'noaa':6s} {B:6d} {f:12.1f} "
                      f"{gp:9.4f} {f/c:9.0f}x")
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
