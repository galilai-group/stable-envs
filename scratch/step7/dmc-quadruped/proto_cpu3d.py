"""Phase-0 design-validating prototype: quadruped-walk through ``stable_envs.render.cpu3d`` with an
ELLIPSOID intersector supplied by MONKEYPATCH (nothing under ``stable_envs/`` is modified).

What is reused verbatim from the shared backend: ``cpu3d.Scene`` (rays as ``compute_ray``, the baked
skybox plate, the finite textured plane under MJWarp's PLANE texgen law, the material/light tables),
``cpu3d.render`` (the megakernel's shading order, spot cone, 0.3 shadow factor, truncating quantiser)
and ``raycast3d.cast_nearest`` / ``any_blocking`` (the shared intersectors). What is added, at runtime
and only here: ``raycast3d.intersect`` learns mjtGeom 4 through ``_ellipsoid`` below;
``Primitives.bound_radius`` learns its bounding sphere ``max(size)``; ``cpu3d.MOVER_KINDS`` admits it.
The intersector is the closed form the design note specifies -- scale the ray into the frame where the
ellipsoid is the unit sphere, solve the unit-sphere quadratic, take the gradient of the implicit
function as the normal -- written elementwise and unrolled as ``raycast3d._mv`` requires.

Variants, so the numbers separate the design from the approximation:
  ``ellipsoid``  the real intersector (the proposed capability)
  ``bsphere``    the ellipsoid drawn as its bounding sphere -- sizes the error an approximation would carry
  ``noell``      the ellipsoid omitted -- what today's cpu3d would draw if it did not refuse
and ``--quant trunc|round`` (cpu3d.QUANT: MJWarp-as-captured truncates; GL rounds).

Reference: the official frames in ``--corpus`` (both mujoco arms), per surface off the segmentation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from stable_envs.render import cpu3d as C3  # noqa: E402
from stable_envs.render import raycast3d as R3  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

ELLIPSOID = 4


# ---------------------------------------------------------------- the proposed intersector (design)
def _ellipsoid(o, d, pos, R, size):
    """Ray / ellipsoid, closed form, in the geom frame scaled to the unit sphere.

    ``size`` (...,3) are the semi-axes. With ``ol = R^T (o - pos)`` and ``dl = R^T d``, the scaled ray
    ``ol' = ol / size``, ``dl' = dl / size`` meets the unit sphere where
    ``|dl'|^2 t^2 + 2 (ol'.dl') t + |ol'|^2 - 1 = 0``; the nearer positive root is the hit and the
    normal is the gradient of ``sum (x_i / s_i)^2`` at the hit, ``p_l / size^2``, normalised and rotated
    back. Every step is elementwise (nine multiplies and six adds per frame change, the quadratic in
    the same association order as ``_sphere``), so the result is batch, order and chunk invariant, and
    a fused twin would be line-for-line the same expressions with ``div_rn``/``sqrt_rn``.
    """
    ol = R3._mtv(R, o - pos)
    dl = R3._mtv(R, d)
    inv = 1.0 / size                                       # (...,3)
    olx, oly, olz = ol[..., 0] * inv[..., 0], ol[..., 1] * inv[..., 1], ol[..., 2] * inv[..., 2]
    dlx, dly, dlz = dl[..., 0] * inv[..., 0], dl[..., 1] * inv[..., 1], dl[..., 2] * inv[..., 2]
    a = dlx * dlx + dly * dly + dlz * dlz
    b = olx * dlx + oly * dly + olz * dlz
    c = olx * olx + oly * oly + olz * olz - 1.0
    disc = b * b - a * c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    a_safe = torch.clamp(a, min=1e-12)
    t = (-b - sq) / a_safe
    t = torch.where(t > 0, t, (-b + sq) / a_safe)
    hit = (disc > 0) & (t > 0) & (a > 1e-12)
    t = torch.where(hit, t, torch.full_like(t, R3.INF))
    tc = t.unsqueeze(-1).clamp(max=1e6)
    pl = ol + tc * dl                                      # hit point, geom frame (unscaled)
    nl = pl * inv * inv                                    # gradient: p_i / s_i^2
    nl = nl / torch.clamp(torch.sqrt(R3._dot(nl, nl)).unsqueeze(-1), min=1e-12)
    return t, R3._mv(R, nl)


_ORIG_INTERSECT = R3.intersect
_ORIG_BOUND = R3.Primitives.bound_radius
_ORIG_MOVER_KINDS = C3.MOVER_KINDS
VARIANT = {"mode": "ellipsoid"}


def _intersect(kind, size, pos, R, o, d):
    if kind == ELLIPSOID:
        if VARIANT["mode"] == "bsphere":
            return R3._sphere(o, d, pos, size.max(-1).values)
        return _ellipsoid(o, d, pos, R, size)
    return _ORIG_INTERSECT(kind, size, pos, R, o, d)


def _bound_radius(self):
    out = _ORIG_BOUND(self)
    s = np.asarray(self.size, np.float64)
    for i, k in enumerate(self.kind):
        if k == ELLIPSOID:
            out[i] = float(s[i].max())
    return out


def patch():
    R3.intersect = _intersect
    R3.Primitives.bound_radius = _bound_radius
    C3.MOVER_KINDS = tuple(_ORIG_MOVER_KINDS) + (ELLIPSOID,)


# ---------------------------------------------------------------- the cell, as a scratch harness
def build(cam: int, res: int, drop_ellipsoid: bool):
    import mujoco
    from dm_control import suite
    env = suite.load("quadruped", "walk", task_kwargs={"random": 0})
    env.reset()
    m = env.physics.model.ptr
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), res)
    movers = [g for g in range(m.ngeom) if int(m.geom_type[g]) != C3.PLANE
              and not (drop_ellipsoid and int(m.geom_type[g]) == ELLIPSOID)]
    sc = C3.Scene(m, cam, res, movers, groups=(0, 1, 2), enable_specular=True, use_shadows=True,
                  device="cpu")
    return m, sc, movers


def poses(m, sc, cam: int, qpos: np.ndarray):
    import mujoco
    d = mujoco.MjData(m)
    B, K = len(qpos), len(sc.movers)
    cp = np.empty((B, 3)); xp = np.empty((B, K, 3)); xm = np.empty((B, K, 3, 3))
    lp = np.empty((B, m.nlight, 3)); ld = np.empty((B, m.nlight, 3))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        cp[i] = d.cam_xpos[cam]
        xp[i] = d.geom_xpos[sc.movers]
        xm[i] = d.geom_xmat[sc.movers].reshape(K, 3, 3)
        lp[i] = d.light_xpos
        ld[i] = d.light_xdir
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    return t(cp), t(xp), t(xm), t(lp), t(ld)


def render(m, sc, cam, qpos, quant: str):
    C3.QUANT = quant
    cp, xp, xm, lp, ld = poses(m, sc, cam, qpos)
    B = len(qpos)
    mat = torch.tensor(np.asarray(m.mat_rgba)[:, :3], dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    ldif = torch.tensor(np.asarray(m.light_diffuse), dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    return C3.render(sc, cp, xp, xm, lp, ld, mat, ldif).numpy(), (cp, xp, xm, lp, ld, mat, ldif)


def per_surface(ours, official, segs):
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    objid, objtype = segs[..., 0], segs[..., 1]
    geom = objtype == 5
    masks = {"sky": objid < 0, "floor": geom & (objid == 0), "body": geom & (objid > 0),
             "ellipsoid": geom & (objid == 3), "site": objtype == 6}
    out = {}
    for k, msk in masks.items():
        n = int(msk.sum())
        out[k] = {"share": float(msk.mean()), "mean_abs": float(d[msk].mean()) if n else None,
                  "exact": float((d[msk] == 0).mean()) if n else None,
                  "within8": float((d[msk] <= 8).mean()) if n else None,
                  "resid_share": float(d[msk].sum() / max(d.sum(), 1))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--sizes", default="84,224")
    ap.add_argument("--modes", default="ellipsoid,bsphere,noell")
    ap.add_argument("--quants", default="trunc,round")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    patch()
    qpos = np.load(f"{a.corpus}/states.npz")["qpos"]
    import mujoco
    results = {"mujoco_local": mujoco.__version__, "torch": torch.__version__, "threads": a.threads,
               "n": int(len(qpos)), "cam": a.cam, "cells": {}}
    for size in (int(s) for s in a.sizes.split(",")):
        refs = {arm: np.load(f"{a.corpus}/official_{arm}_cam{a.cam}_{size}px.npz") for arm in ("mj3110", "mj3120")}
        for mode in a.modes.split(","):
            VARIANT["mode"] = mode
            m, sc, movers = build(a.cam, size, drop_ellipsoid=(mode == "noell"))
            for quant in a.quants.split(","):
                key = f"{size}px_{mode}_{quant}"
                t0 = time.perf_counter()
                ours, _ = render(m, sc, a.cam, qpos, quant)
                wall = time.perf_counter() - t0
                if mode == "ellipsoid" and quant == "trunc":
                    np.savez_compressed(f"{a.out}/proto_{key}.npz", frames=ours)
                    # timing, render only (poses excluded: they are MuJoCo FK, 3-10 us/state)
                    cp, xp, xm, lp, ld, mat, ldif = render(m, sc, a.cam, qpos, quant)[1]
                    for B in (1, 16, 64):
                        reps = 3 if B == 64 else 5
                        t1 = time.perf_counter()
                        for _ in range(reps):
                            C3.render(sc, cp[:B], xp[:B], xm[:B], lp[:B], ld[:B], mat[:B], ldif[:B])
                        results["cells"].setdefault(f"timing_{size}px", {})[f"B{B}_fps_local_1thread"] = \
                            round(B * reps / (time.perf_counter() - t1), 1)
                row = {"wall_s_64_states_incl_fk": round(wall, 2)}
                for arm, ref in refs.items():
                    r = C.pixel_agreement(ours, ref["frames"])
                    r["ssim"] = C.ssim(ours, ref["frames"])
                    r["per_surface"] = per_surface(ours, ref["frames"], ref["segs"])
                    row[f"vs_{arm}"] = r
                    ps = r["per_surface"]
                    print(f"{key} vs {arm}: mean {r['mean_abs']:.3f} exact {r['exact_frac']:.4f} within8 "
                          f"{r['within8_frac']:.4f} ssim {r['ssim']:.4f} max {r['max_abs']} | sky "
                          f"{ps['sky']['mean_abs']:.2f}/{ps['sky']['resid_share']:.2f} floor "
                          f"{ps['floor']['mean_abs']:.2f}/{ps['floor']['resid_share']:.2f} body "
                          f"{ps['body']['mean_abs']:.2f}/{ps['body']['resid_share']:.2f} ell "
                          f"{ps['ellipsoid']['mean_abs']:.2f}/{ps['ellipsoid']['resid_share']:.3f}", flush=True)
                results["cells"][key] = row
        # a contact sheet: official 3.11 / ours (ellipsoid, trunc) / |diff| x8, first 8 states
        try:
            from PIL import Image
            ours = np.load(f"{a.out}/proto_{size}px_ellipsoid_trunc.npz")["frames"]
            ref = refs["mj3110"]["frames"]
            diff = np.clip(np.abs(ours[:8].astype(np.int16) - ref[:8].astype(np.int16)) * 8, 0, 255).astype(np.uint8)
            sheet = np.concatenate([np.concatenate(list(ref[:8]), 1), np.concatenate(list(ours[:8]), 1),
                                    np.concatenate(list(diff), 1)], 0)
            Image.fromarray(sheet).save(f"{a.out}/sheet_{size}px_official311_proto_diffx8.png")
        except Exception as e:  # noqa: BLE001
            print("sheet failed:", e)
    json.dump(results, open(f"{a.out}/proto_results_cam{a.cam}.json", "w"), indent=1)
    print("timing:", {k: v for k, v in results["cells"].items() if k.startswith("timing")})


if __name__ == "__main__":
    main()
