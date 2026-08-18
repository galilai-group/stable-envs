"""Where the fused CarRacing surface disagrees with the per-primitive one, and where its time goes.

The 1000x800 surface is diffed rather than the 96x96 observation: one differing surface pixel is
worth about 1/87 of an output pixel after the resample, so the observation says a pixel is wrong and
the surface says WHICH one and by how much.
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
from stable_envs.envs.carracing import H, KMAX, W, _draw_poly_mask
from stable_envs.render import gfxdraw as G


def capture(env, s, sty, B):
    import stable_envs.envs.carracing as CR
    got = {}
    orig = CR.CarRacing._raster

    def spy(self, B_, gv, gc, go, pv, pc, po, qv, qc, qo, aa):
        got["g"] = (torch.cat(gv, 1), torch.cat(gc, 1), torch.cat(go, 1))
        got["p"] = list(zip(pv, pc, po))
        got["q"] = list(zip(qv, qc, qo))
        got["surface"] = orig(self, B_, gv, gc, go, pv, pc, po, qv, qc, qo, aa)
        return got["surface"]

    CR.CarRacing._raster = spy
    try:
        env._draw(s, sty, B)
    finally:
        CR.CarRacing._raster = orig
    return got


def torch_surface(env, got, s, B):
    dev = env.device
    sty = env.resolve_style(None, None, batch=B)
    cv = G.StyleCanvas(env.gfx, B, torch.zeros(B, 3, device=dev))
    cv.aa = sty["rendering.antialias"] > 0.5
    bi = env.gfx.batch_index(B)
    gv, gc, go = got["g"]
    for i in range(gv.shape[1]):
        G.polygon(cv.use(gc[:, i]).gated(go[:, i]), gv[:, i], None, None, bi, KMAX, True)
    for v, c, g in got["p"]:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    cv.buf.copy_(torch.flip(cv.buf, dims=(1,)))
    for v, c, g in got["q"]:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    env._blit_text(cv.buf, s[:, 26])          # the fused surface is blitted in place too
    return cv.buf, gv.shape[1], len(got["p"]), len(got["q"])


def diag(B=24, seed=11):
    env = se.make("CarRacing", batch=B, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(seed)),
                         device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=B)
    got = capture(env, st, sty, B)
    fused = got["surface"]
    ref, Pg, npre, npost = torch_surface(env, got, st, B)
    d = (fused.to(torch.int16) - ref.to(torch.int16)).abs().amax(-1)
    bad = torch.nonzero(d > 0)
    print(f"Pg={Pg} pre={npre} post={npost}   differing surface pixels: {bad.shape[0]} "
          f"of {d.numel()}  max {int(d.max())}", flush=True)
    for r in bad[:24].tolist():
        b, y, x = r
        print(f"  b={b} y={y} x={x}  fused {fused[b, y, x].tolist()}  ref {ref[b, y, x].tolist()}"
              f"  (pre-flip row {H - 1 - y})", flush=True)
    return env, st, got, fused, ref


def stages(B=256):
    import stable_envs.envs.carracing as CR
    from stable_envs.render import gfxdraw_fused as F
    env = se.make("CarRacing", batch=1, device="cuda")
    pool = env.sample_states(1024)
    st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=B)
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

    o_raster, o_render, o_shrink, o_bin, o_edge = (
        CR.CarRacing._raster, F.render_surface, F.shrink, F.bin_primitives, F.edge_table)
    CR.CarRacing._raster = timed("raster_total", o_raster)
    F.render_surface = timed("  kernel", o_render)
    F.shrink = timed("shrink", o_shrink)
    F.bin_primitives = timed("  bin", o_bin)
    F.edge_table = timed("  edge", o_edge)
    try:
        env._draw(st, sty, B)
        acc.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env._draw(st, sty, B)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        (CR.CarRacing._raster, F.render_surface, F.shrink, F.bin_primitives,
         F.edge_table) = o_raster, o_render, o_shrink, o_bin, o_edge
    print(f"\n=== stages at B={B}: total {1000*total:.2f} ms ===", flush=True)
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {1000*v:8.2f} ms  {100*v/total:5.1f}%", flush=True)


def sweep(B=1024):
    """Tile shape and warp count for the surface kernel, at a batch where the GPU is saturated."""
    from stable_envs.render import gfxdraw_fused as F
    env = se.make("CarRacing", batch=1, device="cuda")
    pool = env.sample_states(1024)
    st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
    print(f"\n=== surface kernel shape sweep at B={B} (ms) ===", flush=True)
    for bins in [(4, 128), (8, 128), (16, 128), (32, 128), (64, 128)]:
        row = [f"bins {bins[0]}x{bins[1]}"]
        for tile, warps in [((1, 128), 1), ((2, 128), 1), ((4, 128), 1)]:
            F.TILE, F.BINS, F.WARPS = tile, bins, warps
            if bins[0] % tile[0] or bins[1] % tile[1]:
                continue
            try:
                for _ in range(2):
                    env.render(st)
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    env.render(st)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                row.append(f"{tile[0]}x{tile[1]}/{warps} {1000*float(np.median(ts)):6.2f}")
            except Exception as e:
                row.append(f"{tile}/{warps} {type(e).__name__}")
        print("  " + "  ".join(row), flush=True)
    F.TILE, F.BINS, F.WARPS = (1, 128), (64, 128), 1


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "diag"):
        diag()
    if what in ("all", "stages"):
        stages()
    if what in ("all", "sweep"):
        sweep()
