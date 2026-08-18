"""Where the time goes in the two families that are slower than reading a stored dataset.

highway-env: split ``render`` into ``scene()`` (python-issued torch ops) and the fused kernel, and
report the primitive count, so the O(pixels x primitives) claim is measured rather than assumed.

CarRacing: stage-by-stage, because its cost is not one kernel -- it is ~200 sequential polygons, each
of which allocates a full-canvas mask and rewrites the whole (B,800,1000,3) buffer, plus a Wu walk
bounded at KMAX=1024 per edge, plus a two-pass resample.
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


def timeit(fn, reps=7, warm=3):
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


def highway(names, batches, size=256):
    from stable_envs.render.raster2d import render_triton
    print("=== highway family ===", flush=True)
    for name in names:
        try:
            env = se.make(name, batch=1, size=size, device="cuda")
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {e}", flush=True)
            continue
        pool = env.sample_states(4096)
        for B in batches:
            st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
            scr = env.screen(size)
            sty = env.resolve_style(None, None, batch=B)
            try:
                sc = env._scene(st, sty, scr)
                out = torch.empty((B, scr.H, scr.W, 3), dtype=torch.uint8, device="cuda")
                t_all = timeit(lambda: env.render(st, out=out))
                t_scene = timeit(lambda: env._scene(st, sty, scr))
                t_ker = timeit(lambda: render_triton(sc, (scr.H, scr.W), 1, out=out))
                px = scr.H * scr.W
                print(f"[{name}] B={B:5d} {scr.H}x{scr.W} N={sc.n_prims:4d} "
                      f"static={sc.static_prefix:4d} | all {1000*t_all:8.2f} ms {B/t_all:9.1f} f/s | "
                      f"scene {1000*t_scene:7.2f} | kernel {1000*t_ker:8.2f} | "
                      f"{B*px*sc.n_prims/t_ker/1e9:6.2f} Gprim-px/s", flush=True)
                del sc, out
            except Exception as e:
                print(f"[{name}] B={B}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
        del env
        torch.cuda.empty_cache()


def carracing(batches):
    print("=== CarRacing ===", flush=True)
    try:
        env = se.make("CarRacing", batch=1, device="cuda")
    except Exception as e:
        print(f"CarRacing unavailable: {type(e).__name__}: {e}", flush=True)
        return
    pool = env.sample_states(1024)
    for B in batches:
        st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
        try:
            t = timeit(lambda: env.render(st), reps=5, warm=2)
            print(f"[CarRacing] B={B:4d} all {1000*t:9.2f} ms {B/t:8.1f} f/s", flush=True)
        except Exception as e:
            print(f"[CarRacing] B={B}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()

    # stage breakdown at a batch that fits comfortably
    B = batches[0]
    st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=B)
    import stable_envs.render.gfxdraw as G
    from stable_envs.envs import carracing as CR

    stage = {}
    orig_polygon, orig_mask, orig_shrink = G.polygon, CR._draw_poly_mask, CR._Shrink.__call__

    def timed(key, fn):
        def wrap(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            stage[key] = stage.get(key, [0, 0.0])
            stage[key][0] += 1
            stage[key][1] += time.perf_counter() - t0
            return r
        return wrap

    G.polygon = timed("gfx_polygon", orig_polygon)
    CR._draw_poly_mask = timed("draw_poly_mask", orig_mask)
    CR._Shrink.__call__ = timed("shrink", orig_shrink)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env._draw(st, sty, B)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        print(f"  stage breakdown at B={B}: total {1000*total:.1f} ms", flush=True)
        for k, (n, s) in sorted(stage.items(), key=lambda kv: -kv[1][1]):
            print(f"    {k:16s} calls {n:5d}  {1000*s:9.2f} ms  {100*s/total:5.1f}%  "
                  f"{1000*s/max(n,1):7.3f} ms/call", flush=True)
    finally:
        G.polygon, CR._draw_poly_mask, CR._Shrink.__call__ = orig_polygon, orig_mask, orig_shrink


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    if which in ("all", "highway"):
        highway(["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"],
                [1024])
    if which in ("all", "carracing"):
        carracing([64, 256])
