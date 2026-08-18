"""Find the bug in the per-primitive windowed cast, without putting it back in the repo.

The bound is proven correct (``bbox_probe.py``: 0 of 36 primitives draw outside their projected box), so
the 52/255 came from the windowing CODE. This reimplements that code standalone, compares its
``(t, normal, index)`` against the full-frame ``_cast`` it must reproduce, and -- the part that names the
bug rather than just detecting it -- reports for each disagreeing ray whether it fell inside any window.

  disagreeing rays all OUTSIDE every window -> the windows do not tile the movers' footprint
  disagreeing rays INSIDE some window        -> the per-window arithmetic or the scatter is wrong
"""
from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raycast3d as R3

NAME, SIZE, B = "DMCBallInCupCatch", 84, 4

env = se.make(NAME, batch=B, size=SIZE, device="cuda")
s = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)),
                    device=env.device, dtype=torch.float32)
sc = env._scene()
ss, R = env.ss, SIZE * env.ss
N = R * R
o, d = sc.cam.rays(R, R, env.device)
oo, dd = o.expand(B, N, 3), d.view(1, N, 3).expand(B, N, 3)
xpos, xmat, sizes, _ = env._fk(s, sc)
P = xpos.shape[1]

ref_t, ref_n, ref_i = env._cast(sc, oo, dd, xpos, xmat, sizes)

# ---- the reverted implementation, verbatim in spirit ----
tile = 64
rad = torch.tensor(sc.prims.bound_radius(), device=env.device, dtype=xpos.dtype)
lo, hi = xpos - rad.view(1, -1, 1), xpos + rad.view(1, -1, 1)
corners = torch.stack(
    [torch.where(torch.tensor([bool(m & 1), bool(m & 2), bool(m & 4)], device=env.device).view(1, 1, 3),
                 hi, lo) for m in range(8)], 2)
ndc = sc.cam.project(corners.reshape(B, P * 8, 3), aspect=1.0).reshape(B, P, 8, -1)
px = (ndc[..., 0] + 1) * 0.5 * SIZE
py = (1 - ndc[..., 1]) * 0.5 * SIZE
x0 = (px.amin(2).floor().long() - 1).clamp(0, SIZE - tile)
y0 = (py.amin(2).floor().long() - 1).clamp(0, SIZE - tile)
T = tile * ss
ar = torch.arange(T, device=env.device)
yy = (y0.view(B, P, 1, 1) * ss + ar.view(1, 1, T, 1)).expand(B, P, T, T)
xx = (x0.view(B, P, 1, 1) * ss + ar.view(1, 1, 1, T)).expand(B, P, T, T)
win = (yy * R + xx).reshape(B, P, -1)

best = torch.full((B, N), R3.INF, device=env.device)
nrm = torch.zeros((B, N, 3), device=env.device)
idx = torch.zeros((B, N), device=env.device, dtype=torch.long)
for i in range(P):
    w = win[:, i]
    w3 = w.unsqueeze(-1).expand(-1, -1, 3)
    t, n = R3.intersect(int(sc.prims.kind[i]), sizes[:, i:i + 1], xpos[:, i:i + 1],
                        xmat[:, i:i + 1], oo.gather(1, w3), dd.gather(1, w3))
    cur = best.gather(1, w)
    upd = t < cur
    best.scatter_(1, w, torch.where(upd, t, cur))
    nrm.scatter_(1, w3, torch.where(upd.unsqueeze(-1), n.expand(B, w.shape[1], 3), nrm.gather(1, w3)))
    idx.scatter_(1, w, torch.where(upd, torch.full_like(w, i), idx.gather(1, w)))

st = env._static_t(sc, ss).view(1, N).expand(B, N)
hit_ref, hit_win = ref_t < st, best < st
bad = (hit_ref != hit_win) | (hit_ref & hit_win & (ref_i != idx))
print(f"{NAME}: {P} prims, tile {tile}, {int(bad.sum())} disagreeing rays of {B * N}")

if int(bad.sum()):
    inside = torch.zeros((B, N), device=env.device, dtype=torch.bool)
    for i in range(P):
        inside.scatter_(1, win[:, i], torch.ones_like(win[:, i], dtype=torch.bool))
    nb, nin = int(bad.sum()), int((bad & inside).sum())
    print(f"  inside some window: {nin}/{nb}   outside every window: {nb - nin}/{nb}")
    b, f = torch.nonzero(bad, as_tuple=True)
    for k in range(min(5, len(b))):
        bb, ff = int(b[k]), int(f[k])
        print(f"   ray (b={bb}, y={ff // R}, x={ff % R})  ref t={float(ref_t[bb, ff]):.4f} "
              f"i={int(ref_i[bb, ff])} | win t={float(best[bb, ff]):.4f} i={int(idx[bb, ff])} | "
              f"inside={bool(inside[bb, ff])}")
    # Do the windows even cover the pixels the reference draws?
    miss = int((hit_ref & ~inside).sum())
    print(f"  reference-drawn pixels NOT covered by any window: {miss}")
