"""The numbers behind CarRacing's ceiling: how much is drawn, and how fast is it drawn.

Two questions the wall clock alone cannot answer:

  * how many primitives does a pixel ACTUALLY have to evaluate, after binning? If that number is
    large the fix is a better bound; if it is small the kernel is already doing the minimum work the
    composite defines and the ceiling is arithmetic.
  * do the remaining torch stages scale with the batch? A stage that is flat in the batch is
    dispatch and can be fused away; one that scales is real work.
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


def main(B=1024):
    grabbed = {}
    o_render, o_edge = F.render_surface, F.edge_table

    def spy(vert, meta, edge, aa, H, W, PG, PPOST, out=None):
        grabbed.update(meta=meta, PG=PG, P=meta.shape[1], vert=vert, edge=edge, aa=aa)
        return o_render(vert, meta, edge, aa, H, W, PG, PPOST, out)

    F.render_surface = spy
    try:
        env = se.make("CarRacing", batch=1, device="cuda")
        pool = env.sample_states(1024)
        st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
        env.render(st)
    finally:
        F.render_surface = o_render

    meta, P, PG = grabbed["meta"], grabbed["P"], grabbed["PG"]
    lst, cnt, nbx = F.bin_primitives(meta[..., F.PM_X0:F.PM_Y1 + 1], CR.H, CR.W, *F.BINS)
    c = cnt.float()
    bh, bw = F.TILE
    binh, binw = F.BINS
    print(f"primitives per frame: {P} total, {PG} gfxdraw quads", flush=True)
    print(f"bins {binh}x{binw}: {cnt.shape[1]} per frame; list length mean {float(c.mean()):.2f} "
          f"median {float(c.median()):.0f} p95 {float(c.quantile(0.95)):.0f} "
          f"max {int(cnt.max())}", flush=True)
    per_px = float(c.mean())
    npx = B * CR.H * CR.W
    print(f"surface pixels per batch: {npx/1e6:.0f} M; primitive-pixel evaluations "
          f"{npx * per_px / 1e9:.1f} G", flush=True)

    # the kernel alone, and the same kernel with an empty list, so the difference is the drawing
    t_k = timeit(lambda: o_render(grabbed["vert"], meta, grabbed["edge"], grabbed["aa"],
                                  CR.H, CR.W, PG, P))
    empty = meta.clone()
    empty[..., F.PM_X0] = 1
    empty[..., F.PM_Y0] = 1
    empty[..., F.PM_X1] = 0
    empty[..., F.PM_Y1] = 0
    t_n = timeit(lambda: o_render(grabbed["vert"], empty, grabbed["edge"], grabbed["aa"],
                                  CR.H, CR.W, PG, P))
    draw = t_k - t_n
    evals = npx * per_px
    print(f"surface kernel {1000*t_k:.2f} ms; empty-list {1000*t_n:.2f} ms; drawing "
          f"{1000*draw:.2f} ms", flush=True)
    print(f"drawing rate: {evals/draw/1e9:.1f} G primitive-pixel evaluations/s", flush=True)

    # which stages are dispatch (flat in batch) and which are work (linear in batch)
    print("\nstage scaling (ms):", flush=True)
    for b2 in (128, 512, 1024):
        st2 = torch.as_tensor(pool[np.arange(b2) % len(pool)], device="cuda")
        sty = env.resolve_style(None, None, batch=b2)
        s = st2.to(torch.float64)
        acc = {}

        def timed(key, fn):
            def wrap(*a, **k):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = fn(*a, **k)
                torch.cuda.synchronize()
                acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
                return r
            return wrap

        F.edge_table = timed("edge_table", o_edge)
        F.render_surface = timed("kernel", o_render)
        try:
            env._draw(s, sty, b2)
            acc.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            env._draw(s, sty, b2)
            torch.cuda.synchronize()
            tot = time.perf_counter() - t0
        finally:
            F.edge_table, F.render_surface = o_edge, o_render
        print(f"  B={b2:5d} total {1000*tot:7.2f}  edge_table {1000*acc['edge_table']:6.2f}  "
              f"kernel {1000*acc['kernel']:7.2f}", flush=True)
        del st2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    main()
