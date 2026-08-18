"""Correctness and tile-shape sweep for the binned rasterizer.

Correctness first: every registered env that builds a ``Scene2D`` is rendered both ways and diffed,
at ss=1 and ss=2. Then the tile shape is swept, because "compact tiles win" is a claim about a curve
whose shape depends on the frame aspect ratio and the primitive count, and the highway family spans
256x1024 at 188 primitives to 256x256 at 436.
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


def scene_of(env, batch, seed=0):
    st = env.sample_states(batch, rng=np.random.default_rng(seed))
    try:
        return env.scene(torch.as_tensor(st, device=env.device, dtype=torch.float32))
    except Exception:
        return None


def size_of(env):
    return (env.H, env.W) if hasattr(env, "H") and hasattr(env, "W") else env.size


def correctness(only=None):
    print("=== culled vs full, every registered scene ===", flush=True)
    bad = []
    for name in sorted(se.REGISTRY):
        if only and name not in only:
            continue
        try:
            env = se.make(name, batch=8, device="cuda")
            sc = scene_of(env, 8)
            if sc is None:
                continue
            sz = size_of(env)
            for ss in (1, 2):
                if ss == 2 and (sz if isinstance(sz, int) else max(sz)) > 512:
                    continue
                a = R.render_triton(sc, sz, supersample=ss, cull=False).clone()
                b = R.render_triton(sc, sz, supersample=ss, cull=True)
                d = int((a.to(torch.int16) - b.to(torch.int16)).abs().max())
                if d:
                    npx = int(((a.to(torch.int16) - b.to(torch.int16)).abs() > 0).any(-1).sum())
                    bad.append((name, ss, d, npx))
                    print(f"  MISMATCH {name} ss={ss}: max {d}/255 on {npx} px", flush=True)
            print(f"  ok {name:28s} N={sc.n_prims:4d} {sz}", flush=True)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}", flush=True)
        finally:
            torch.cuda.empty_cache()
    print(f"=== {len(bad)} mismatching cells ===", flush=True)
    return bad


def sweep(names, B=1024, size=256):
    print("\n=== tile shape sweep (kernel only, ms) ===", flush=True)
    shapes = [(1, 128, 4), (16, 16, 4), (16, 32, 4), (32, 16, 4), (32, 32, 4), (32, 32, 8),
              (16, 64, 4), (16, 64, 8), (64, 32, 8), (32, 64, 8), (8, 128, 4), (8, 128, 8)]
    for name in names:
        try:
            env = se.make(name, batch=1, size=size, device="cuda")
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {e}", flush=True)
            continue
        st = torch.as_tensor(env.sample_states(B), device="cuda")
        sty = env.resolve_style(None, None, batch=B)
        scr = env.screen(size) if hasattr(env, "screen") else None
        sc = env._scene(st, sty, scr) if scr is not None else env.scene(st)
        hw = (scr.H, scr.W) if scr is not None else size_of(env)
        out = torch.empty((B, hw[0] if not isinstance(hw, int) else hw,
                           hw[1] if not isinstance(hw, int) else hw, 3),
                          dtype=torch.uint8, device="cuda")
        base = timeit(lambda: R.render_triton(sc, hw, 1, out=out, cull=False))
        row = [f"[{name}] N={sc.n_prims:4d} {hw} full {1000*base:7.2f}"]
        best = None
        for bh, bw, wp in shapes:
            R._CULL_TILE = (bh, bw, wp)
            try:
                t = timeit(lambda: R.render_triton(sc, hw, 1, out=out, cull=True))
            except Exception as e:
                row.append(f"{bh}x{bw}/{wp}: {type(e).__name__}")
                continue
            row.append(f"{bh}x{bw}/{wp} {1000*t:6.2f}")
            if best is None or t < best[0]:
                best = (t, (bh, bw, wp))
        R._CULL_TILE = (32, 32, 8)
        print("  ".join(row), flush=True)
        print(f"    best {best[1]} {1000*best[0]:.2f} ms  speedup {base/best[0]:.2f}x", flush=True)
        del sc, out, env
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    if what in ("all", "correct"):
        correctness(only)
    if what in ("all", "sweep"):
        sweep(["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"])
