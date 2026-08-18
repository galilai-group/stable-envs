"""Per-pass timing of ``DMCBase._render_group`` / ``GymMujoco._render_group`` at 256 px.

``scratch/census/dmc_passes.py`` does this at 84 px and batch 256, where the mover passes still
dominate. At 256 px the two FULL-FRAME passes (shadow from static, reflection) grow 9x while the
windowed mover passes do not grow at all, so the ranking is a different one and the profile has to be
retaken at the operating point rather than extrapolated.

Runs one CHUNK (``RAY_BUDGET`` worth of environments) rather than the whole batch, because the passes
are per-chunk and the ranking is what is wanted, not the wall clock.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raycast3d as R3

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="DMCBallInCupCatch,DMCCartpoleBalance,DMCReacherEasy,"
                                 "DMCFingerTurnEasy,DMCPointMassEasy,MujocoPusher,MujocoReacher")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()


def sync_time(fn, n):
    for _ in range(1):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


print(f"{torch.cuda.get_device_name(0)}  {a.size}px  chunk b{a.batch}\n")
for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    B, SIZE = a.batch, a.size
    env = se.make(name, batch=B, size=SIZE, device="cuda")
    s = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)),
                        device=env.device, dtype=torch.float32)
    sc = env._scene(SIZE)
    ss = env.ss
    R = SIZE * ss
    N = R * R
    dev = env.device
    o, d = sc.cam.rays(R, R, dev)
    oo = o.expand(B, N, 3)
    dd = d.view(1, N, 3).expand(B, N, 3)
    fk = env._fk(s, sc)
    if len(fk) == 4:
        xpos, xmat, sizes, mat_rgb = fk
    else:
        xpos, xmat = fk
        P0 = xpos.shape[1]
        sizes = (torch.tensor(sc.prims.size, device=dev, dtype=torch.float32).view(1, P0, 3)
                 .expand(B, P0, 3))
        mat_rgb = torch.tensor(sc.prims.mat_rgb, device=dev, dtype=torch.float32).view(1, P0, 3)
    P = xpos.shape[1]
    L = sc.lights.to(dev)
    sl = sc.shadow_light
    st = env._static_t(sc, ss).view(1, N).expand(B, N)

    total = sync_time(lambda: env._render_group(*( (sc, s, SIZE, ss) if len(fk) == 4 else
                                                   (sc, s, SIZE, ss,
                                                    torch.ones(B, 3, device=dev),
                                                    torch.ones(B, 1, device=dev)))), a.reps)
    t_fk = sync_time(lambda: env._fk(s, sc), a.reps)
    t_widx = sync_time(lambda: env._mover_idx(sc, xpos, SIZE, ss), a.reps)
    widx = env._mover_idx(sc, xpos, SIZE, ss)
    nwin = -1 if widx is None else int(widx.shape[1])
    if widx is None:
        t_cast = sync_time(lambda: env._cast(sc, oo, dd, xpos, xmat, sizes), a.reps)
        t_m, n_m, i_m = env._cast(sc, oo, dd, xpos, xmat, sizes)
        pm = oo + t_m.clamp(max=1e6).unsqueeze(-1) * dd
        i_w, pm_w = i_m, pm
    else:
        w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        oo_w, dd_w = oo.gather(1, w3), dd.gather(1, w3)
        t_cast = sync_time(lambda: env._cast(sc, oo_w, dd_w, xpos, xmat, sizes), a.reps)
        t_w, n_w, i_w = env._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
        pm_w = oo_w + t_w.clamp(max=1e6).unsqueeze(-1) * dd_w
        t_m = torch.full((B, N), R3.INF, device=dev).scatter_(1, widx, t_w)
    cand = None
    if sl >= 0 and hasattr(env, "_shadow_candidates"):
        cand = env._shadow_candidates(sc, ss)
        cand = cand[0] if isinstance(cand, tuple) else cand
    sel = cand if cand is not None else torch.arange(N, device=dev)
    ps = oo[:, sel] + st[:, sel].clamp(max=1e6).unsqueeze(-1) * dd[:, sel]
    t_occ_m = sync_time(lambda: env._occluded(sc, pm_w, L, sl, xpos, xmat, sizes, skip=i_w,
                                              eps=sc.eps), a.reps) if sl >= 0 and P > 1 else 0.0
    t_occ_s = sync_time(lambda: env._occluded(sc, ps, L, sl, xpos, xmat, sizes,
                                              eps=sc.eps), a.reps) if sl >= 0 else 0.0
    t_refl = 0.0
    if getattr(sc, "reflectance", 0) and sc.reflectance > 0:
        base = torch.zeros(B, N, 3, device=dev)
        bi = torch.arange(B, device=dev).unsqueeze(1)
        tex_rgb = torch.tensor(sc.prims.tex_rgb, device=dev, dtype=torch.float32)
        spec = torch.tensor(sc.prims.spec, device=dev, dtype=torch.float32)
        shin = torch.tensor(sc.prims.shininess, device=dev, dtype=torch.float32)
        t_refl = sync_time(lambda: env._reflect(sc, base, oo, dd, st, xpos, xmat, sizes, mat_rgb,
                                                tex_rgb, spec, shin, L, ss, SIZE, bi), a.reps)

    cover = (t_m < st).float().mean().item()
    print(f"\n{name}  P={P}  {SIZE}px ss={ss}  chunk b{B}  window={nwin} rays of {N}  "
          f"total {total * 1e3:8.1f} ms")
    acc = 0.0
    for lbl, t in (("_fk (kinematics)", t_fk), ("_mover_idx (window)", t_widx),
                   ("primary mover cast", t_cast), ("shadow from movers", t_occ_m),
                   ("shadow from STATIC", t_occ_s), ("reflection", t_refl)):
        acc += t
        print(f"    {lbl:22s} {t * 1e3:8.1f} ms  {100 * t / total:5.1f}%")
    print(f"    {'accounted':22s} {acc * 1e3:8.1f} ms  {100 * acc / total:5.1f}%  "
          f"(rest = shading + composite + box filter)")
    mir = env._mirror_idx(sc, xpos, SIZE, ss) if hasattr(env, "_mirror_idx") and \
        getattr(sc, "reflectance", 0) else None
    print(f"    movers cover {100 * cover:.2f}% of pixels; shadow candidates "
          f"{sel.numel()}/{N} ({100 * sel.numel() / N:.1f}%); mirror window "
          f"{'full frame' if mir is None else f'{mir.shape[1]}/{N}'}")
    del env
    torch.cuda.empty_cache()
