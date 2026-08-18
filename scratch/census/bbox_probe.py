"""Does a primitive's projected bounding box actually contain the pixels that primitive draws?

The per-primitive window failed at 52/255 on ``ball_in_cup`` -- clipped geometry -- and was reverted
without finding out why. This answers the question directly and without that code: render full-frame,
ask ``_cast`` which primitive won each pixel, and compare each primitive's TRUE pixel extent against the
box ``bound_radius`` + ``cam.project`` predicts for it.

If a primitive's true extent escapes its predicted box, the bound is wrong and the window was never
going to work. If every extent fits, the bound is right and the bug was in the windowing code -- two
very different repairs, and the difference is one measurement.
"""
from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raycast3d as R3

NAME = "DMCBallInCupCatch"
SIZE, B = 84, 8

env = se.make(NAME, batch=B, size=SIZE, device="cuda")
s = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)),
                    device=env.device, dtype=torch.float32)
sc = env._scene()
ss = env.ss
R = SIZE * ss
N = R * R
o, d = sc.cam.rays(R, R, env.device)
oo, dd = o.expand(B, N, 3), d.view(1, N, 3).expand(B, N, 3)
xpos, xmat, sizes, _ = env._fk(s, sc)
P = xpos.shape[1]

t_m, _, i_m = env._cast(sc, oo, dd, xpos, xmat, sizes)
st = env._static_t(sc, ss).view(1, N).expand(B, N)
hit = t_m < st                                   # (B,N) pixels this primitive set actually draws

# predicted boxes, exactly as prim_windows computed them
rad = torch.tensor(sc.prims.bound_radius(), device=env.device, dtype=xpos.dtype)
lo, hi = xpos - rad.view(1, -1, 1), xpos + rad.view(1, -1, 1)
corners = torch.stack(
    [torch.where(torch.tensor([bool(m & 1), bool(m & 2), bool(m & 4)], device=env.device).view(1, 1, 3),
                 hi, lo) for m in range(8)], 2)
ndc = sc.cam.project(corners.reshape(B, P * 8, 3), aspect=1.0).reshape(B, P, 8, -1)
px = (ndc[..., 0] + 1) * 0.5 * SIZE
py = (1 - ndc[..., 1]) * 0.5 * SIZE
bx0, bx1 = px.amin(2).floor() - 1, px.amax(2).ceil() + 1
by0, by1 = py.amin(2).floor() - 1, py.amax(2).ceil() + 1

yy = (torch.arange(R, device=env.device) // ss).view(R, 1).expand(R, R).reshape(-1)
xx = (torch.arange(R, device=env.device) // ss).view(1, R).expand(R, R).reshape(-1)

print(f"{NAME}: {P} primitives, {SIZE}px, batch {B}")
print(f"{'prim':>5s} {'kind':>5s} {'drawn px':>9s} {'true box (x,y)':>22s} {'predicted box':>22s}  ok")
bad = 0
for i in range(P):
    m = hit & (i_m == i)
    n = int(m.sum())
    if n == 0:
        continue
    b, flat = torch.nonzero(m, as_tuple=True)
    tx0, tx1 = int(xx[flat].min()), int(xx[flat].max())
    ty0, ty1 = int(yy[flat].min()), int(yy[flat].max())
    # the predicted box has to hold for EVERY batch row, so take the loosest true / tightest predicted
    p0x, p1x = float(bx0[:, i].min()), float(bx1[:, i].max())
    p0y, p1y = float(by0[:, i].min()), float(by1[:, i].max())
    ok = (tx0 >= p0x) and (tx1 <= p1x) and (ty0 >= p0y) and (ty1 <= p1y)
    bad += not ok
    if not ok or i < 4:
        print(f"{i:5d} {int(sc.prims.kind[i]):5d} {n:9d} "
              f"{f'({tx0},{tx1})x({ty0},{ty1})':>22s} "
              f"{f'({p0x:.0f},{p1x:.0f})x({p0y:.0f},{p1y:.0f})':>22s}  {'OK' if ok else 'ESCAPES'}")
print(f"\n{bad} of {P} primitives draw outside their predicted box")
print(f"union of predicted boxes: x {float(bx0.min()):.0f}..{float(bx1.max()):.0f}  "
      f"y {float(by0.min()):.0f}..{float(by1.max()):.0f}  (frame is 0..{SIZE})")
print(f"movers cover {100*hit.float().mean():.2f}% of pixels")
