"""Which primitive the bounding-box cull drops that the full kernel draws.

Each primitive is rendered ALONE, both ways, so the answer is the primitive index and its geometry
rather than a pixel coordinate. A primitive that differs alone is one whose box is not a bound.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raster2d as R
from stable_envs.render.scene import Scene2D


def main(name="Intersection", B=64, seed=0):
    env = se.make(name, batch=1, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(seed)), device="cuda")
    sc = env.scene(st)
    hw = (env.H, env.W)
    a = R.render_triton(sc, hw, env.ss, cull=False)
    b = R.render_triton(sc, hw, env.ss, cull=True)
    d = (a.to(torch.int16) - b.to(torch.int16)).abs().amax(-1)
    print(f"{name} {hw} N={sc.n_prims} B={B}: differing pixels {int((d > 0).sum())} "
          f"max {int(d.max())}", flush=True)
    if not int(d.max()):
        return
    bad = torch.nonzero(d > 0)
    print(f"  first: {bad[:5].tolist()}", flush=True)
    rows = sorted({int(r[0]) for r in bad})
    print(f"  envs affected: {rows[:10]} ({len(rows)})", flush=True)
    e0 = rows[0]
    one = Scene2D(kind=sc.kind[e0:e0 + 1].clone(), geom=sc.geom[e0:e0 + 1].clone(),
                  fill=sc.fill[e0:e0 + 1].clone(), line=sc.line[e0:e0 + 1].clone(),
                  line_w=sc.line_w[e0:e0 + 1].clone(), background=sc.background[e0:e0 + 1].clone(),
                  world=sc.world, flip_y=sc.flip_y)
    for i in range(sc.n_prims):
        k = torch.zeros_like(one.kind)
        k[0, i] = one.kind[0, i]
        s1 = Scene2D(kind=k, geom=one.geom, fill=one.fill, line=one.line, line_w=one.line_w,
                     background=one.background, world=one.world, flip_y=one.flip_y)
        x = R.render_triton(s1, hw, env.ss, cull=False)
        y = R.render_triton(s1, hw, env.ss, cull=True)
        m = int((x.to(torch.int16) - y.to(torch.int16)).abs().max())
        if m:
            g = one.geom[0, i].tolist()
            v = [(g[2 * j], g[2 * j + 1]) for j in range(one.geom.shape[2] // 2)]
            area2 = sum(v[j][0] * v[(j + 1) % len(v)][1] - v[(j + 1) % len(v)][0] * v[j][1]
                        for j in range(len(v)))
            print(f"  prim {i} kind={int(one.kind[0, i])} lw={float(one.line_w[0, i]):.4f} "
                  f"maxdiff {m}  verts {v}  shoelace {area2!r}", flush=True)


if __name__ == "__main__":
    B = int(os.environ.get("PROBE_B", "1024"))
    for n in (sys.argv[1:] or ["Intersection", "Racetrack"]):
        main(n, B=B)
