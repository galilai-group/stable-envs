"""What would a CORRECT CarRacing port actually run at?

The reference draws its polygons strictly in order and each ``aapolygon`` blends against whatever is
already on the canvas, so an exact port cannot batch across polygons: it needs one sequential step
per polygon.  probe_census_font measured the number of steps (median 81, worst 118 after the
reference's own cull) and that one batched ``aaline`` call costs ~8 ms of python dispatch regardless
of how many edges it covers.  This times the whole loop, eager and CUDA-graph-captured, so the
"speedup is gone" claim can be checked against a number instead of an estimate.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from stable_envs.render import gfxdraw as G

H, W = 800, 1000
STEPS = 118                      # worst-case primitives surviving the reference's cull


def build(B, dev="cuda"):
    g = G.Gfx(H, W, dev)
    cv = g.canvas(B, (102, 204, 102))
    col = g.color((102, 230, 102))
    gen = torch.Generator(device=dev).manual_seed(0)
    ctr = torch.rand((STEPS, B, 1, 2), generator=gen, device=dev, dtype=torch.float64) * 900 + 50
    loc = torch.tensor([[-90., -90.], [90., -90.], [90., 90.], [-90., 90.]],
                       dtype=torch.float64, device=dev).view(1, 1, 4, 2)
    verts = torch.trunc(ctr + loc).to(torch.int64)                # (STEPS,B,4,2)
    bi4 = g.batch_index(B).view(B, 1).expand(B, 4).reshape(-1)
    return g, cv, col, verts, bi4


def one_step(cv, col, v, bi4, B):
    """aapolygon (all four edges in ONE call) then filled_polygon -- the reference's own order."""
    vx, vy = v[..., 0], v[..., 1]
    nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
    G.aaline(cv, vx.reshape(-1), vy.reshape(-1), nx.reshape(-1), ny.reshape(-1), col, 200, bi4)
    cv.paint(G.fill_polygon_mask(vx, vy, cv.ys, cv.xs, cv.W), (102, 230, 102))


def bench(fn, iters=3):
    for _ in range(1):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print(f"=== {STEPS} sequential aapolygon+fill steps at 1000x800 (official = 174.5 f/s) ===")
    for B in (8, 32, 128):
        g, cv, col, verts, bi4 = build(B)

        def run():
            for s in range(STEPS):
                one_step(cv, col, verts[s], bi4, B)
        dt = bench(run)
        print(f"  eager  B={B:4d}: {dt:7.3f} s/call -> {B/dt:9.1f} f/s "
              f"({1000*dt/STEPS:6.2f} ms per polygon)")
        del g, cv, verts
        torch.cuda.empty_cache()

    print("\n=== the same loop captured as a CUDA graph (python dispatch removed) ===")
    for B in (8, 32, 128):
        g, cv, col, verts, bi4 = build(B)

        def run():
            for s in range(STEPS):
                one_step(cv, col, verts[s], bi4, B)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            gr = torch.cuda.CUDAGraph()
            t0 = time.perf_counter()
            with torch.cuda.graph(gr):
                run()
            cap = time.perf_counter() - t0
            gr.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                gr.replay()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3
            print(f"  graph  B={B:4d}: {dt:7.3f} s/replay -> {B/dt:9.1f} f/s "
                  f"({1000*dt/STEPS:6.2f} ms per polygon; capture took {cap:.1f} s)")
        except Exception as e:
            print(f"  graph  B={B:4d}: capture failed: {type(e).__name__}: {e}")
        del g, cv, verts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
