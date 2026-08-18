"""What the raster2d tiling costs the envs that do NOT cull.

Below ``_CULL_MIN_PRIMS`` the bounding-box pre-pass is skipped entirely, so the only change those
envs see is the new pixel-to-program mapping: a 2-D tile instead of a run of consecutive addresses.
That is free in principle and has to be measured, because the tile is chosen to divide the frame
width and a frame whose width is not a friendly number gets a different block count.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from scratch.fastraster.orig_raster import render_orig
from stable_envs.render import raster2d as R


def timeit(fn, reps=11, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(B=1024):
    print(f"{'env':22s} {'frame':12s} {'N':>4s} {'orig ms':>9s} {'new ms':>9s} {'ratio':>7s} cull",
          flush=True)
    for name in sorted(se.REGISTRY):
        try:
            env = se.make(name, batch=1, device="cuda")
            st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)), device="cuda")
            sc = env.scene(st)
        except Exception:
            continue
        try:
            hw = (env.H, env.W) if hasattr(env, "H") and hasattr(env, "W") else env.size
            out = torch.empty((B, *(hw if not isinstance(hw, int) else (hw, hw)), 3),
                              dtype=torch.uint8, device="cuda")
            a = timeit(lambda: render_orig(sc, hw, env.ss))
            b = timeit(lambda: R.render_triton(sc, hw, env.ss, out=out))
            d = (render_orig(sc, hw, env.ss).to(torch.int16)
                 - R.render_triton(sc, hw, env.ss).to(torch.int16)).abs().max()
            print(f"{name:22s} {str(hw):12s} {sc.n_prims:4d} {1000*a:9.2f} {1000*b:9.2f} "
                  f"{a/b:7.2f}x {'yes' if sc.n_prims >= R._CULL_MIN_PRIMS else 'no ':3s} "
                  f"maxdiff {int(d)}", flush=True)
        except Exception as e:
            print(f"{name:22s} {type(e).__name__}: {e}", flush=True)
        finally:
            del env, sc
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    main()
