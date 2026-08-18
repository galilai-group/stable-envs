"""Time each of DM Control's four ray passes separately. Aim the next change instead of guessing.

``dmcontrol.py`` already records the shape of the problem in its own source: three of the passes are
FULL-FRAME, and the movers occupy 0.2-3.0% of the frame. What it does not record is how the cost splits
between them, and that decides which one to bound first -- bounding the primary cast is easy and
bounding the static shadow is not, so it matters whether the easy one is 10% or 60%.

Each pass is timed with an explicit sync so the number is GPU time rather than queue-fill time.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raycast3d as R3

CASES = ["DMCBallInCupCatch", "DMCCartpoleBalance", "DMCReacherEasy"]
B, SIZE = 256, 84


def sync_time(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


for name in CASES:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    env = se.make(name, batch=B, size=SIZE, device="cuda")
    s = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)),
                        device=env.device, dtype=torch.float32)
    sc = env._scene()
    ss = env.ss
    R = SIZE * ss
    N = R * R
    dev = env.device
    o, d = sc.cam.rays(R, R, dev)
    oo = o.expand(B, N, 3)
    dd = d.view(1, N, 3).expand(B, N, 3)
    xpos, xmat, sizes, mat_rgb = env._fk(s, sc)
    P = xpos.shape[1]
    L = sc.lights.to(dev)
    sl = sc.shadow_light
    st = env._static_t(sc, ss).view(1, N).expand(B, N)

    total = sync_time(lambda: env.render(s))
    t_fk = sync_time(lambda: env._fk(s, sc))
    t_cast = sync_time(lambda: env._cast(sc, oo, dd, xpos, xmat, sizes))
    t_m, n_m, i_m = env._cast(sc, oo, dd, xpos, xmat, sizes)
    pm = oo + t_m.clamp(max=1e6).unsqueeze(-1) * dd
    ps = oo + st.clamp(max=1e6).unsqueeze(-1) * dd
    t_occ_m = sync_time(lambda: env._occluded(sc, pm, L, sl, xpos, xmat, sizes, skip=i_m,
                                              eps=sc.eps)) if sl >= 0 and P > 1 else 0.0
    t_occ_s = sync_time(lambda: env._occluded(sc, ps, L, sl, xpos, xmat, sizes,
                                              eps=sc.eps)) if sl >= 0 else 0.0

    # How much of the frame do the movers actually cover? That is the ceiling on what bounding buys.
    cover = (t_m < st).float().mean().item()
    print(f"\n{name}  P={P} prims  {SIZE}px b{B}  total {total*1e3:7.1f} ms = {B/total:8,.0f} f/s")
    for lbl, t in (("_fk (kinematics)", t_fk), ("primary mover cast", t_cast),
                   ("shadow from movers", t_occ_m), ("shadow from STATIC", t_occ_s)):
        print(f"    {lbl:22s} {t*1e3:7.1f} ms  {100*t/total:5.1f}%")
    acc = t_fk + t_cast + t_occ_m + t_occ_s
    print(f"    {'accounted':22s} {acc*1e3:7.1f} ms  {100*acc/total:5.1f}%   "
          f"(rest is reflection + shading + the box filter)")
    print(f"    movers cover {100*cover:.2f}% of pixels -> bounding the mover passes can remove "
          f"at most {100*(1-cover):.2f}% of their work")
    del env
    torch.cuda.empty_cache()
