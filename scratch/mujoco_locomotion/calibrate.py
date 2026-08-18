"""Where the mover-pixel residual lives: multisampling, or the shading law?

Three questions, each with a number:

  1. Score our movers against a reference rendered at ``offsamples=1``. If the residual is MuJoCo's
     4x multisampling bleeding across the silhouette, this is where it goes away — and the answer
     bounds how much of the gap is recoverable by fitting at all.
  2. Fit (ambient, diffuse, specular, exponent) by least squares on mover-INTERIOR pixels and report
     how far the fit is from the nominal (0.1, 0.4, 0.25, 64) and what it buys. A fit that lands on
     the nominal says the law is right and the residual is elsewhere.
  3. Throughput, measured properly: warmed up, over enough frames, and against the official renderer
     driven the same way.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import gymnasium as gym                                                     # noqa: E402
import mujoco                                                              # noqa: E402
import stable_envs as se                                                    # noqa: E402
from stable_envs.envs import mujoco_locomotion as ML                        # noqa: E402
from stable_envs.testing import conformance as C                            # noqa: E402
from scratch.mujoco_locomotion.parity import erode, mover_mask              # noqa: E402


def reference_at_offsamples(cls, size, offsamples):
    """A fresh official env whose offscreen buffer is built at the requested sample count.

    ``offsamples`` is read when the ``MjrContext`` is created, which happens at the first render, so
    it has to be set before that and cannot be changed on a live viewer.
    """
    env = gym.make(cls.ENV_ID, render_mode="rgb_array", width=int(size), height=int(size))
    u = env.unwrapped
    u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), int(size))
    u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), int(size))
    u.model.vis.quality.offsamples = int(offsamples)
    env.reset(seed=0)
    u.render()
    return env


def fit_law(env, cls, ref, states, size, mask):
    """Least squares for (A, D, S, exponent) on mover-interior pixels.

    The model is ``pixel = mat*(A + D*ndl) + S*ndh^p`` per channel, which is linear in (A, D, S) for
    a fixed ``p``, so ``p`` is swept and the linear part solved exactly at each value. Everything is
    evaluated at the SAME geometry our renderer uses, so a difference is the law and not the cast.
    """
    dev = env.device
    sc = env._scene(size, None)
    ss = 1                                   # per-pixel, so a fitted level is not an edge average
    R = size * ss
    N = R * R
    o, d = sc.cam.rays(R, R, dev)
    L = sc.lights.to(dev)
    official = cls.reference_render(ref, states).astype(np.float64) / 255.0
    rows_m, rows_ndl, rows_ndh, rows_y = [], [], [], []
    for k, st in enumerate(states):
        xpos, xmat = env._fk(torch.as_tensor(st[None], device=dev, dtype=torch.float32), sc)
        P = xpos.shape[1]
        sizes = torch.tensor(sc.prims.size, device=dev, dtype=torch.float32).view(1, P, 3)
        oo, dd = o.expand(1, N, 3), d.view(1, N, 3)
        t, n, i = env._cast(sc, oo, dd, xpos, xmat, sizes.expand(1, P, 3))
        p = oo + t.clamp(max=1e6).unsqueeze(-1) * dd
        v = -dd
        li, att = ML._light_geometry(L, 0, p)
        ndl = torch.clamp(ML._dot(n, li), min=0)
        h = ML._norm(li + v)
        ndh = torch.clamp(ML._dot(n, h), min=0)
        mat = torch.tensor(sc.prims.mat_rgb, device=dev, dtype=torch.float32)[i]
        m = torch.as_tensor(mask[k].reshape(-1), device=dev)
        sel = m & (t < R3_INF)
        if not bool(sel.any()):
            continue
        rows_m.append(mat[0][sel[0]].cpu().numpy())
        rows_ndl.append(ndl[0][sel[0]].cpu().numpy())
        rows_ndh.append(ndh[0][sel[0]].cpu().numpy())
        rows_y.append(official[k].reshape(-1, 3)[sel[0].cpu().numpy()])
    if not rows_m:
        return None
    M = np.concatenate(rows_m)                       # (n,3)
    NDL = np.concatenate(rows_ndl)                   # (n,)
    NDH = np.concatenate(rows_ndh)
    Y = np.concatenate(rows_y)                       # (n,3)
    best = None
    for p in (16, 32, 48, 64, 80, 96, 128):
        A = np.stack([M.reshape(-1), (M * NDL[:, None]).reshape(-1),
                      np.repeat(NDH ** p, 3)], 1)
        coef, *_ = np.linalg.lstsq(A, Y.reshape(-1), rcond=None)
        res = float(np.sqrt(np.mean((A @ coef - Y.reshape(-1)) ** 2)))
        if best is None or res < best[0]:
            best = (res, p, [float(c) for c in coef])
    nominal_A = np.stack([M.reshape(-1), (M * NDL[:, None]).reshape(-1),
                          np.repeat(NDH ** 64, 3)], 1)
    nominal = nominal_A @ np.array([0.1, 0.4, 0.5 * 0.5 * ML.SPECULAR_SCALE])
    return {"n_pixels": int(len(Y)), "best_rmse": round(best[0], 6), "best_exponent": best[1],
            "best_ADS": [round(c, 5) for c in best[2]],
            "nominal_ADS": [0.1, 0.4, round(0.25 * ML.SPECULAR_SCALE, 5)], "nominal_exponent": 64,
            "nominal_rmse": round(float(np.sqrt(np.mean((nominal - Y.reshape(-1)) ** 2))), 6)}


R3_INF = 1e30


def bench(cls, size, batch, iters=3):
    env = cls(batch=batch, size=size, device="cuda")
    st = env.sample_states(batch, rng=np.random.default_rng(0))
    env.render(st[:4])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        env.render(st)
    torch.cuda.synchronize()
    return round(batch * iters / (time.perf_counter() - t0), 1)


def bench_reference(cls, size, n=200):
    ref = C.reference_env(cls, size)
    u = ref.unwrapped
    st = cls(batch=1, size=size, device="cuda").sample_states(n, rng=np.random.default_rng(0))
    u.set_state(st[0].astype(np.float64), np.zeros(u.model.nv))
    a = u.render().copy()
    t0 = time.perf_counter()
    for s in st:
        u.set_state(s.astype(np.float64), np.zeros(u.model.nv))
        f = u.render()
    dt = time.perf_counter() - t0
    moved = int(np.abs(f.astype(np.int16) - a.astype(np.int16)).max())
    return round(n / dt, 1), moved


def run(name, size=240, n=24):
    cls = se.REGISTRY[name]
    env = cls(batch=n, size=size, device="cuda")
    ref4 = C.reference_env(cls, size)
    states = env.sample_states(n, rng=np.random.default_rng(0))
    ours = env.render(states).cpu().numpy()
    mm = mover_mask(cls, ref4, states, size)
    mi = erode(mm)
    rec = {"env": name, "size": size, "n": n}

    for offs in (4, 1):
        r = ref4 if offs == 4 else reference_at_offsamples(cls, size, offs)
        official = cls.reference_render(r, states)
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        rec[f"offsamples{offs}"] = {
            "mover_interior_exact": round(float((d[mi] == 0).mean()), 4),
            "mover_interior_within1": round(float((d[mi] <= 1).mean()), 4),
            "mover_interior_within2": round(float((d[mi] <= 2).mean()), 4),
            "mover_interior_mean": round(float(d[mi].mean()), 4),
            "mover_exact": round(float((d[mm] == 0).mean()), 4),
            "all_exact": round(float((d == 0).mean()), 6)}

    if len(cls._bundle(size)[2].light_diffuse) == 0:            # single-light domains only
        rec["fit"] = fit_law(env, cls, ref4, states[:8], size, mi[:8])

    # Does the FITTED specular lobe (a broader, weaker highlight, which is what per-vertex Gouraud
    # interpolation of a per-pixel one looks like) buy enough to be worth a constant that departs
    # from MuJoCo's documented law?
    official = cls.reference_render(ref4, states)
    base_shade = ML._shade
    rec["lobe"] = {}
    for sspec, sexp in ((1.0, 1.0), (0.988, 1.0), (0.85, 0.75), (0.708, 0.5), (0.5, 0.35)):
        def variant(p, n, v, L, mat, spec, shininess, unlit=None, _s=sspec, _e=sexp):
            return base_shade(p, n, v, L, mat, spec * _s / ML.SPECULAR_SCALE, shininess * _e, unlit)
        ML._shade = variant
        f = env.render(states).cpu().numpy()
        d = np.abs(f.astype(np.int16) - official.astype(np.int16)).max(-1)
        rec["lobe"][f"spec{sspec}_exp{sexp}"] = {
            "mover_interior_within1": round(float((d[mi] <= 1).mean()), 4),
            "mover_interior_exact": round(float((d[mi] == 0).mean()), 4),
            "mover_interior_mean": round(float(d[mi].mean()), 4),
            "all_exact": round(float((d == 0).mean()), 6),
            "mean_abs": round(float(d.mean()), 5)}
    ML._shade = base_shade
    rec["fps_ours_240_b256"] = bench(cls, 240, 256)
    rec["fps_ours_84_b1024"] = bench(cls, 84, 1024)
    rec["fps_ours_480_b64"] = bench(cls, 480, 64)
    fps, moved = bench_reference(cls, 240)
    rec["fps_reference_240"] = fps
    rec["reference_frames_differ_by"] = moved
    return rec


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ML.ENVS)
    out = []
    for nm in names:
        try:
            r = run(nm)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"env": nm, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
