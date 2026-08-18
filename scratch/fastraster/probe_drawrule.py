"""The ``pygame.draw.polygon`` rule, both implementations, printed term by term at one pixel."""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.carracing import _draw_poly_mask

PIX_Y, PIX_X, ENV = 224, 519, 0


def kernel_rule(vx, vy, Y, X, verbose=False):
    """What ``_surface_kernel`` computes: lt / le / nact and the even-k range test."""
    nv = len(vx)
    miny, maxy = min(vy), max(vy)
    lt = le = nact = 0
    extra = False
    for e in range(nv):
        j = (e + nv - 1) % nv
        ax, ay, bx, by = vx[j], vy[j], vx[e], vy[e]
        up = ay < by
        y1, y2 = min(ay, by), max(ay, by)
        x1, x2 = (ax, bx) if up else (bx, ax)
        act = (ay != by) and (((y1 <= Y) and (Y < y2)) or ((Y == maxy) and (y2 == maxy)))
        q = np.float32((Y - y1) * (x2 - x1)) / np.float32(max(y2 - y1, 1))
        xc = int(math.floor(q) if nact % 2 == 0 else math.ceil(q)) + x1
        if verbose:
            print(f"      e={e} a=({ax},{ay}) b=({bx},{by}) act={act} rank={nact} q={q} xc={xc}")
        if act:
            lt += xc < X
            le += xc <= X
            nact += 1
        if ay == by and miny < ay < maxy and Y == ay and min(ax, bx) <= X <= max(ax, bx):
            extra = True
    ka, kb = max(lt - 1, 0), min(le - 1, nact - 2)
    hit = (ka <= kb) and (ka % 2 == 0 or ka + 1 <= kb)
    if miny == maxy and Y == miny and min(vx) <= X <= max(vx):
        hit = True
    return hit or extra, (lt, le, nact, ka, kb, extra)


def torch_rule(vx, vy, Y, X):
    scr = torch.tensor([[[float(a), float(b)] for a, b in zip(vx, vy)]], dtype=torch.float64,
                       device="cuda")
    ys = torch.tensor([Y], device="cuda")
    xs = torch.tensor([X], device="cuda")
    return bool(_draw_poly_mask(scr, ys, xs)[0, 0, 0])


def main(B=24, seed=11):
    from scratch.fastraster.diag_carracing import capture
    env = se.make("CarRacing", batch=B, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(seed)),
                         device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=B)
    got = capture(env, st, sty, B)
    for k, (v, c, g) in enumerate(got["p"]):
        vi = torch.trunc(v[ENV]).to(torch.int64).cpu().numpy()
        vx, vy = list(vi[:, 0]), list(vi[:, 1])
        mine, terms = kernel_rule(vx, vy, PIX_Y, PIX_X)
        theirs = torch_rule(vx, vy, PIX_Y, PIX_X)
        flag = "  <-- DISAGREE" if mine != theirs else ""
        print(f"  pre[{k}] gate={bool(g[ENV])} colour={c[ENV].tolist()} nv={len(vx)} "
              f"kernel={mine} torch={theirs} terms={terms}{flag}", flush=True)
        if mine != theirs:
            print(f"    verts {list(zip(vx, vy))}", flush=True)
            kernel_rule(vx, vy, PIX_Y, PIX_X, verbose=True)


if __name__ == "__main__":
    main()
