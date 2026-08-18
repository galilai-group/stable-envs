"""Bisect the primitive list to find the one whose fused draw disagrees with the reference.

Both paths are run on the same captured lists truncated to a prefix, so the first prefix at which
they differ names the primitive. Cheaper and more decisive than reasoning about which rule it is.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.carracing import KMAX, _draw_poly_mask
from stable_envs.render import gfxdraw as G


def run_pair(env, got, B, n_gfx, n_pre, n_post, aa):
    gv, gc, go = got["g"]
    pre, post = got["p"][:n_pre], got["q"][:n_post]
    fused = env._raster(B, [gv[:, :n_gfx]], [gc[:, :n_gfx]], [go[:, :n_gfx]],
                        [p[0] for p in pre], [p[1] for p in pre], [p[2] for p in pre],
                        [q[0] for q in post], [q[1] for q in post], [q[2] for q in post], aa)
    cv = G.StyleCanvas(env.gfx, B, torch.zeros(B, 3, device=env.device))
    cv.aa = aa
    bi = env.gfx.batch_index(B)
    for i in range(n_gfx):
        G.polygon(cv.use(gc[:, i]).gated(go[:, i]), gv[:, i], None, None, bi, KMAX, True)
    for v, c, g in pre:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    cv.buf.copy_(torch.flip(cv.buf, dims=(1,)))
    for v, c, g in post:
        cv.use(c).gated(g).paint(_draw_poly_mask(v, cv.ys, cv.xs))
    d = (fused.to(torch.int16) - cv.buf.to(torch.int16)).abs().amax(-1)
    return int((d > 0).sum()), torch.nonzero(d > 0)[:4]


def main(B=24, seed=11):
    from scratch.fastraster.diag_carracing import capture
    env = se.make("CarRacing", batch=B, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(seed)),
                         device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=B)
    aa = sty["rendering.antialias"] > 0.5
    got = capture(env, st, sty, B)
    ng, npre, npost = got["g"][0].shape[1], len(got["p"]), len(got["q"])
    print(f"gfx={ng} pre={npre} post={npost}", flush=True)
    for tag, args in (("gfx only", (ng, 0, 0)), ("gfx+pre", (ng, npre, 0)),
                      ("all", (ng, npre, npost))):
        n, where = run_pair(env, got, B, *args, aa)
        print(f"  {tag:10s} differing pixels {n}  {where.tolist()}", flush=True)
    # narrow the gfx prefix
    lo, hi = 0, ng
    while lo < hi:
        mid = (lo + hi) // 2
        n, _ = run_pair(env, got, B, mid, 0, 0, aa)
        if n:
            hi = mid
        else:
            lo = mid + 1
    print(f"  first bad gfx prefix: {lo} (primitive index {lo - 1})", flush=True)
    if lo <= ng:
        gv, gc, go = got["g"]
        i = lo - 1
        print(f"  colour {gc[:, i][0].tolist()} verts {torch.trunc(gv[:, i])[0].tolist()} "
              f"gate {go[:, i][:4].tolist()}", flush=True)
        n, where = run_pair(env, got, B, lo, 0, 0, aa)
        print(f"  pixels {n} at {where.tolist()}", flush=True)


if __name__ == "__main__":
    main()
