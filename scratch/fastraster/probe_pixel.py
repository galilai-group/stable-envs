"""Which primitive, which edge, and which of the two rules writes the one disagreeing pixel.

The kernel's anti-aliasing rule is re-evaluated in numpy for one pixel and every edge, next to what
``gfxdraw.aaline`` actually writes there. One of the two makes a write the other does not, and this
prints the edge and the case it took rather than leaving that to be inferred from a frame diff.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.carracing import H, W
from stable_envs.render import gfxdraw_fused as F

PIX_Y, PIX_X, ENV = 224, 519, 0            # pre-flip row, column, batch row


def kernel_rule(eg, Y, X):
    """What ``_surface_kernel`` does for one edge at one pixel: (case, weight or 'solid')."""
    (draw, sx, sy, xx0, yy0, dx, dy, xdir, kind, major, E, lo, bx0, by0, bx1, by1) = eg
    if not draw:
        return None
    if (Y == sy) and (X == sx):
        return ("solid-start", 256)
    if kind == F.K_VERT:
        if dy > 0 and X == xx0 and yy0 <= Y <= yy0 + dy:
            return ("vline", 256)
        return None
    if kind == F.K_HORIZ:
        if Y == yy0 and lo <= X <= lo + dx:
            return ("hline", 256)
        return None
    if kind == F.K_GEN + 4:                                  # y-major
        kk = Y - yy0
        if not (1 <= kk < major):
            return None
        acc = kk * E
        wraps, wgt = acc >> 16, (acc & 0xFFFF) >> 8
        xa = xx0 + xdir * wraps
        if X == xa:
            return ("wu-a", 255 - wgt)
        if X == xa + xdir:
            return ("wu-b", wgt)
        return None
    kk = (X - xx0) * xdir                                    # x-major
    if not (1 <= kk < major):
        return None
    acc = kk * E
    wraps, wgt = acc >> 16, (acc & 0xFFFF) >> 8
    ya = yy0 + wraps
    if Y == ya:
        return ("wu-a", 255 - wgt)
    if Y == ya + 1:
        return ("wu-b", wgt)
    return None


def reference_writes(vx, vy, Y, X):
    """Every ``aaline`` write that lands on (Y, X), by replaying the walk. gfxdraw rule 4."""
    from stable_envs.render import gfxdraw as G
    out = []
    n = len(vx)
    for e in range(n):
        a = torch.tensor([vx[e]], dtype=torch.int64)
        b = torch.tensor([vy[e]], dtype=torch.int64)
        c = torch.tensor([vx[(e + 1) % n]], dtype=torch.int64)
        d = torch.tensor([vy[(e + 1) % n]], dtype=torch.int64)
        draw, x1, y1, x2, y2 = G.clip_line(a, b, c, d, W, H)
        if not bool(draw[0]):
            continue
        x1, y1, x2, y2 = (int(t[0]) for t in (x1, y1, x2, y2))
        if (y1, x1) == (Y, X):
            out.append((e, "solid-start", 256))
        flip = y1 > y2
        xx0, yy0 = (x2, y2) if flip else (x1, y1)
        dxs = (x1 if flip else x2) - xx0
        dy = (y1 if flip else y2) - yy0
        xdir = 1 if dxs >= 0 else -1
        dx = abs(dxs)
        if dx == 0:
            if dy > 0 and X == xx0 and yy0 <= Y <= yy0 + dy:
                out.append((e, "vline", 256))
            continue
        if dy == 0:
            lo = min(xx0, xx0 + xdir * dx)
            if Y == yy0 and lo <= X <= lo + dx:
                out.append((e, "hline", 256))
            continue
        ymaj = dy > dx
        major, minor = (dy, dx) if ymaj else (dx, dy)
        E = (minor << 16) // max(major, 1)
        for kk in range(1, min(major, 1025)):
            acc = kk * E
            wraps, wgt = acc >> 16, (acc & 0xFFFF) >> 8
            if ymaj:
                pts = [((yy0 + kk, xx0 + xdir * wraps), 255 - wgt),
                       ((yy0 + kk, xx0 + xdir * (wraps + 1)), wgt)]
            else:
                pts = [((yy0 + wraps, xx0 + xdir * kk), 255 - wgt),
                       ((yy0 + wraps + 1, xx0 + xdir * kk), wgt)]
            for (py, px), w in pts:
                if (py, px) == (Y, X):
                    out.append((e, f"wu k={kk}", w))
    return out


def main():
    from scratch.fastraster.diag_carracing import capture
    env = se.make("CarRacing", batch=24, device="cuda")
    st = torch.as_tensor(env.sample_states(24, rng=np.random.default_rng(11)),
                         device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=24)
    got = capture(env, st, sty, 24)
    gv, gc, go = got["g"]
    vi = torch.trunc(gv).to(torch.int64)[ENV].cpu().numpy()        # (Pg,4,2)
    cols = gc[ENV].cpu().numpy()
    gates = go[ENV].cpu().numpy()
    eg = F.edge_table(torch.trunc(gv[ENV, :, :, 0]).to(torch.int64).cpu(),
                      torch.trunc(gv[ENV, :, :, 1]).to(torch.int64).cpu(), H, W).numpy()
    print(f"pixel pre-flip (y={PIX_Y}, x={PIX_X}), env {ENV}, {vi.shape[0]} gfx quads", flush=True)
    for i in range(vi.shape[0]):
        if not gates[i]:
            continue
        vx, vy = vi[i, :, 0], vi[i, :, 1]
        mine = [(e, *r) for e in range(4) if (r := kernel_rule(eg[i, e], PIX_Y, PIX_X))]
        theirs = reference_writes(list(vx), list(vy), PIX_Y, PIX_X)
        if mine or theirs:
            print(f"  prim {i} colour {cols[i]} verts {list(zip(vx, vy))}", flush=True)
            print(f"    kernel : {mine}", flush=True)
            print(f"    gfxdraw: {theirs}", flush=True)


if __name__ == "__main__":
    main()
