"""What is CarRacing's floor, and how much of it is drawing?

The observation is defined as ``pygame.transform.smoothscale`` of a FIXED 1000x800 surface, and that
filter reads every one of those 800,000 pixels exactly once. So there is a cost floor that has
nothing to do with how cleverly the polygons are rasterized, and the useful question is not "how fast
is the renderer" but "how much of the wall time is already the floor".

Three measurements, all at the same batch:

  1. the whole render, as shipped;
  2. the two resample passes alone, on a surface that already exists -- the part no rasterizer can
     remove;
  3. a NULL rasterizer: the same kernel launched with an empty primitive list, i.e. the cost of
     writing an 800,000-pixel surface per environment and nothing else.

(2) + (3) is the floor of any implementation that honours the contract. Whatever is left is drawing,
and only that part is worth optimizing.
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
import stable_envs.envs.carracing as CR
from stable_envs.render import gfxdraw_fused as F


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


def stages(env, st, B):
    """Per-stage wall time inside one ``_draw``, by wrapping the pieces."""
    sty = env.resolve_style(None, None, batch=B)
    s = st.to(torch.float64)
    acc = {}
    orig = {"raster": CR.CarRacing._raster, "render": F.render_surface, "shrink": F.shrink,
            "bin": F.bin_primitives, "edge": F.edge_table, "text": CR.CarRacing._blit_text}

    def timed(key, fn):
        def wrap(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
            return r
        return wrap

    CR.CarRacing._raster = timed("pack+kernel", orig["raster"])
    F.render_surface = timed("  surface kernel", orig["render"])
    F.shrink = timed("resample", orig["shrink"])
    F.bin_primitives = timed("  bin", orig["bin"])
    F.edge_table = timed("  edge_table", orig["edge"])
    CR.CarRacing._blit_text = timed("text", orig["text"])
    try:
        env._draw(s, sty, B)
        acc.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env._draw(s, sty, B)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        (CR.CarRacing._raster, F.render_surface, F.shrink, F.bin_primitives, F.edge_table,
         CR.CarRacing._blit_text) = (orig["raster"], orig["render"], orig["shrink"], orig["bin"],
                                     orig["edge"], orig["text"])
    print(f"  total {1000*total:8.2f} ms  ({B/total:9.1f} f/s)", flush=True)
    named = sum(v for k, v in acc.items() if not k.startswith("  "))
    for k, v in sorted(acc.items(), key=lambda kv: (kv[0].startswith("  "), -kv[1])):
        print(f"    {k:18s} {1000*v:8.2f} ms  {100*v/total:5.1f}%", flush=True)
    print(f"    {'geometry assembly':18s} {1000*(total-named):8.2f} ms  "
          f"{100*(total-named)/total:5.1f}%", flush=True)


def floor(env, B):
    """The two costs the contract makes unavoidable: write the surface, then box-filter it."""
    dev = env.device
    surf = torch.zeros((B, CR.H, CR.W, 3), dtype=torch.uint8, device=dev)

    def resample():
        x = F.shrink(surf.reshape(B * CR.H, CR.W, 3), env.sx)
        return F.shrink(x.reshape(B, CR.H, 96 * 3), env.sy)

    t_res = timeit(resample)

    # a null rasterizer: one primitive, gated off, so every tile walks an empty list and stores
    vert = torch.zeros((B, 1, 8, 2), dtype=torch.int32, device=dev)
    meta = torch.zeros((B, 1, F.PM_W), dtype=torch.int32, device=dev)
    meta[..., F.PM_NV] = 4
    meta[..., F.PM_X0] = 1                    # empty box -> nothing is ever drawn
    meta[..., F.PM_Y0] = 1
    edge = torch.zeros((B, 1, 4, F.EG_W), dtype=torch.int32, device=dev)
    aa = torch.zeros(B, dtype=torch.int32, device=dev)
    t_null = timeit(lambda: F.render_surface(vert, meta, edge, aa, CR.H, CR.W, 1, 1))
    print(f"  null rasterizer (surface written, nothing drawn) {1000*t_null:8.2f} ms  "
          f"{B/t_null:9.1f} f/s", flush=True)
    print(f"  resample only (surface already exists)           {1000*t_res:8.2f} ms  "
          f"{B/t_res:9.1f} f/s", flush=True)
    print(f"  contract floor = null + resample                 {1000*(t_null+t_res):8.2f} ms  "
          f"{B/(t_null+t_res):9.1f} f/s", flush=True)
    del surf, vert, meta, edge
    torch.cuda.empty_cache()


def main(B=1024):
    env = se.make("CarRacing", batch=1, device="cuda")
    pool = env.sample_states(1024)
    st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
    print(f"=== CarRacing at batch {B}: surface {CR.W}x{CR.H} = "
          f"{B * CR.H * CR.W / 1e6:.0f} M pixels per batch, output 96x96 ===", flush=True)
    stages(env, st, B)
    floor(env, B)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    main()
