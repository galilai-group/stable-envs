"""Is the exact-AA path bandwidth-bound or launch-bound? Capture it in a CUDA graph and find out.

The AA pass costs ~30 tiny kernels per polygon edge (Cohen-Sutherland runs six fixed iterations of
~20 elementwise ops on (B,) tensors, then two gather/blend/scatters). Those kernels do almost no
work, so at small batch the renderer is paying launch latency, not memory bandwidth. The render is
shape-static for a fixed B, so the whole thing captures into a single graph -- which turns the
question into a measurement instead of an assertion.
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
import gpu_render as R

CASES = [("CartPole-v1", R.CartPoleGPU, 2), ("Pendulum-v1", R.PendulumGPU, 1),
         ("Acrobot-v1", R.AcrobotGPU, 2), ("MountainCar-v0", R.MountainCarGPU, 1)]


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024, 4096, 8192])
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'env':16s} {'mode':12s} {'B':>6s} {'frames/s':>12s} {'Gpix/s':>9s} {'speedup':>9s}")
    print("-" * 70)
    for name, cls, dim in CASES:
        for aa in (True, False):
            rend = cls(device="cuda", aa=aa)
            px = cls.H * cls.W
            for B in args.batches:
                g = np.random.default_rng(0)
                st = torch.as_tensor(g.uniform(-1.0, 1.0, (B, dim)), dtype=torch.float64,
                                     device="cuda")
                try:
                    dt = timed(lambda: rend.render(st), args.reps)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                eager = B * args.reps / dt

                # CUDA-graph capture: same code, shapes fixed, launch cost paid once
                try:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(3):
                            rend.render(st)
                    torch.cuda.current_stream().wait_stream(s)
                    gph = torch.cuda.CUDAGraph()
                    pool = torch.cuda.graph_pool_handle()
                    with torch.cuda.graph(gph, pool=pool):
                        out = rend.render(st)
                    dt = timed(gph.replay, args.reps)
                    graph = B * args.reps / dt
                    assert out.shape[0] == B
                except Exception as e:                             # noqa: BLE001
                    graph = float("nan")
                    print(f"    graph capture failed at B={B}: {type(e).__name__}: {e}")
                print(f"{name if B == args.batches[0] and aa else '':16s} "
                      f"{('aa' if aa else 'noaa'):>4s}{'':8s}{B:6d} {eager:12.1f} "
                      f"{eager*px/1e9:9.4f} {'eager':>9s}")
                print(f"{'':16s} {('aa' if aa else 'noaa')+'+graph':12s} {B:6d} {graph:12.1f} "
                      f"{graph*px/1e9:9.4f} {graph/eager:8.2f}x")
                del gph
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
