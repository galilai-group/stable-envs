"""Does the bounding box and the bin list actually reach the tile that owns the disagreeing pixel?"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
import stable_envs.envs.carracing as CR
from stable_envs.render import gfxdraw_fused as F

OUT_Y, OUT_X, ENV = 575, 519, 0


def main(B=24, seed=11):
    grabbed = {}
    orig = F.render_surface

    def spy(vert, meta, edge, aa, H, W, PG, PPOST, out=None):
        grabbed.update(vert=vert, meta=meta, PG=PG, PPOST=PPOST)
        return orig(vert, meta, edge, aa, H, W, PG, PPOST, out)

    F.render_surface = spy
    try:
        env = se.make("CarRacing", batch=B, device="cuda")
        st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(seed)), device="cuda")
        env.render(st)
    finally:
        F.render_surface = orig

    meta, vert = grabbed["meta"], grabbed["vert"]
    PG, PPOST = grabbed["PG"], grabbed["PPOST"]
    P = meta.shape[1]
    print(f"P={P} PG={PG} PPOST={PPOST} meta {tuple(meta.shape)}", flush=True)
    box = meta[ENV, :, F.PM_X0:F.PM_Y1 + 1].cpu().numpy()
    nv = meta[ENV, :, F.PM_NV].cpu().numpy()
    bh, bw = F.TILE
    binh, binw = F.BINS
    tiy, tix = OUT_Y // bh, OUT_X // bw
    ty0, ty1, tx0, tx1 = tiy * bh, tiy * bh + bh - 1, tix * bw, tix * bw + bw - 1
    print(f"tile rows {ty0}..{ty1} cols {tx0}..{tx1}; bin ({ty0 // binh}, {tx0 // binw})", flush=True)
    lst, cnt, nbx = F.bin_primitives(meta[..., F.PM_X0:F.PM_Y1 + 1], CR.H, CR.W, binh, binw)
    bid = (ty0 // binh) * nbx + (tx0 // binw)
    ids = lst[ENV, bid, :int(cnt[ENV, bid])].cpu().numpy().tolist()
    print(f"bin holds {len(ids)} primitives: {ids}", flush=True)
    for i in range(PG, P):
        x0, y0, x1, y1 = box[i]
        reach = (x1 >= tx0) and (x0 <= tx1) and (y1 >= ty0) and (y0 <= ty1)
        vs = vert[ENV, i, :nv[i]].cpu().numpy()
        print(f"  prim {i} nv={nv[i]} box=({x0},{y0},{x1},{y1}) reaches_tile={reach} "
              f"in_bin={i in ids} vy={list(vs[:, 1])} vx={list(vs[:, 0])}", flush=True)


if __name__ == "__main__":
    main()
