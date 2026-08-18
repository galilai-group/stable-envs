"""Fused CarRacing against the frozen frames, then against pygame, then timed.

The frozen frames come from ``dump_ref.py`` and were produced by the per-primitive torch path, which
itself measured 100.0000% exact against pygame. So a byte-for-byte match here is a match against the
official renderer, and it is checked with and without the visited-tile style because that mask is the
one variation of this env that reaches a pixel.
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

REF = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/se_ref"


def cmp(tag, got, want):
    g, w = got.cpu(), want.cpu()
    d = (g.to(torch.int16) - w.to(torch.int16)).abs()
    npx = int((d > 0).any(-1).sum())
    print(f"  {tag:14s} max {int(d.max()):3d}  px {npx}/{d.numel()//3}  "
          f"exact {100*(1-npx/(d.numel()//3)):.4f}%", flush=True)
    return int(d.max()) == 0


def correctness():
    blob = torch.load(f"{REF}/CarRacing.pt", weights_only=True)
    st = blob["states"].cuda()
    env = se.make("CarRacing", batch=st.shape[0], device="cuda")
    ok = cmp("default", env.render(st), blob["frame"])
    ok &= cmp("styled", env.render(st, style=blob["style"].cuda()), blob["frame_style"])
    # a second, larger pool so the check is not one seed
    pool = torch.as_tensor(env.sample_states(24, rng=np.random.default_rng(11)), device="cuda")
    e2 = se.make("CarRacing", batch=24, device="cuda")
    ref = _torch_path(e2, pool)
    ok &= cmp("pool24", e2.render(pool), ref)
    print(f"=== {'MATCH' if ok else 'MISMATCH'} ===", flush=True)
    return ok


def _torch_path(env, states):
    """The per-primitive path this replaces, rebuilt here from the pieces that still exist."""
    from stable_envs.envs.carracing import H, KMAX, W, _draw_poly_mask
    from stable_envs.render import gfxdraw as G
    s = states.to(torch.float64)
    B = s.shape[0]
    sty = env.resolve_style(None, None, batch=B)
    dev = env.device
    cv = G.StyleCanvas(env.gfx, B, torch.zeros(B, 3, device=dev))
    cv.aa = sty["rendering.antialias"] > 0.5
    bi = env.gfx.batch_index(B)
    gfx, pre, post = _capture(env, s, sty, B)
    for v, c, g in zip(*gfx):
        G.polygon(cv.use(c).gated(g), v, None, None, bi, KMAX, True)
    for v, c, g in pre:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    cv.buf.copy_(torch.flip(cv.buf, dims=(1,)))
    for v, c, g in post:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    env._blit_text(cv.buf, s[:, 26])
    return env.sy(env.sx(cv.buf, 2), 1)


def _capture(env, s, sty, B):
    """Run ``_draw``'s assembly and hand back the polygon lists instead of the frame."""
    import stable_envs.envs.carracing as CR
    grabbed = {}
    orig = CR.CarRacing._raster

    def spy(self, B_, gv, gc, go, pv, pc, po, qv, qc, qo, aa):
        grabbed["g"] = (torch.cat(gv, 1), torch.cat(gc, 1), torch.cat(go, 1))
        grabbed["p"] = list(zip(pv, pc, po))
        grabbed["q"] = list(zip(qv, qc, qo))
        return orig(self, B_, gv, gc, go, pv, pc, po, qv, qc, qo, aa)

    CR.CarRacing._raster = spy
    try:
        env._draw(s, sty, B)
    finally:
        CR.CarRacing._raster = orig
    gv, gc, go = grabbed["g"]
    n = gv.shape[1]
    return (([gv[:, i] for i in range(n)], [gc[:, i] for i in range(n)],
             [go[:, i] for i in range(n)]), grabbed["p"], grabbed["q"])


def speed(batches=(64, 256, 1024)):
    print("\n=== throughput at 96x96 ===", flush=True)
    env = se.make("CarRacing", batch=1, device="cuda")
    pool = env.sample_states(1024)
    for B in batches:
        st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
        try:
            for _ in range(3):
                env.render(st)
            torch.cuda.synchronize()
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                env.render(st)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = float(np.median(ts))
            print(f"  B={B:5d}  {1000*t:9.2f} ms  {B/t:10.1f} f/s", flush=True)
        except Exception as e:
            print(f"  B={B}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "correct"):
        correctness()
    if what in ("all", "speed"):
        speed()
