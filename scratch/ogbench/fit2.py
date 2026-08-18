"""Break the ambient/diffuse degeneracy, then find out why wall tops still miss.

The floor alone cannot separate ambient from diffuse (its ``n.l`` is 1 everywhere), so the first fit split
1.489 evenly between them and predicted 114 instead of 15 on vertical faces. Fitting on floor pixels
(n.l = 1) TOGETHER WITH wall-side pixels (n.l = 0) identifies both.

Then the open question: floor pixels match the per-pixel model to 96% exact, but wall-top pixels -- same
normal, same view ray, same functional form -- only reach ~25%. The hypothesis is that MuJoCo lights
per VERTEX and lets the rasterizer interpolate, so a box face (4 vertices, 4x4 world units across) gets a
bilinear approximation of a ``(n.h)^60`` lobe while the plane, which MuJoCo subdivides finely, does not.
That is testable: fit each top face's observed colour as a bilinear function of the face's own (u,v) and
look at the residual. If it collapses to ~0, the shading is Gouraud and reproducing it exactly means
evaluating S at the face corners and interpolating, not evaluating it per pixel.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from general_check import make2, agent_geoms  # noqa: E402
from analytic_rgb import base_rgb  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF


def run(loco, size, img=224):
    import mujoco
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    for _ in range(20):
        env.step(u.action_space.sample())
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    obs = r.render().astype(np.float64)

    gid, t, fwd, o, dr = RC.cast(m, d, cam, img, img)
    gid = np.where(t >= INF / 2, -1, gid)
    p = o + t[..., None] * dr
    n = normals(m, d, gid, p)
    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    L = np.array([0.0, 0.0, 1.0])
    hv = L + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)
    ndl = np.clip(n @ L, 0, None)
    ndh = np.clip((n * hv).sum(-1), 0, None)
    ag, names = agent_geoms(m)
    base = np.zeros((img, img, 3))
    for g in np.unique(gid):
        if g < 0:
            continue
        base[gid == g] = base_rgb(m, int(g))

    floor = gid == 0
    wall = np.isin(gid, [g for g in range(m.ngeom) if (names[g] or "").startswith("block_")])
    wtop = wall & (n[..., 2] > 0.5)
    wside = wall & (n[..., 2] <= 0.5)
    agm = np.isin(gid, ag)
    b = np.zeros(gid.shape, bool)
    b[1:, :] |= gid[1:, :] != gid[:-1, :]
    b[:-1, :] |= gid[1:, :] != gid[:-1, :]
    b[:, 1:] |= gid[:, 1:] != gid[:, :-1]
    b[:, :-1] |= gid[:, 1:] != gid[:, :-1]

    res = dict(env=f"{loco}maze-{size}")

    # ---- joint fit on floor (n.l=1) + wall sides (n.l=0), unsaturated only ---
    Sobs = np.where((base.max(-1) > 1e-6)[..., None],
                    obs / np.maximum(base * 255.0, 1e-9), np.nan).mean(-1)
    sat = (obs >= 254.5).any(-1)
    fitmask = (floor | wside) & ~b & ~sat & np.isfinite(Sobs)
    best = None
    for pexp in range(1, 129):
        A = np.stack([np.ones(fitmask.sum()), ndl[fitmask], ndh[fitmask] ** pexp], 1)
        coef, *_ = np.linalg.lstsq(A, Sobs[fitmask], rcond=None)
        rms = float(np.sqrt(((A @ coef - Sobs[fitmask]) ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, pexp, coef.tolist())
    rms, pexp, (ca, cd, cs) = best
    res["fit"] = dict(ambient=round(ca, 6), diffuse=round(cd, 6), specular=round(cs, 6),
                      exponent=pexp, rms_S=round(rms, 6), n_fit=int(fitmask.sum()))

    S = ca + cd * ndl + cs * ndh ** pexp
    pred = np.clip(np.round(np.clip(base * S[..., None], 0, 1) * 255.0), 0, 255)
    dd = np.abs(pred - obs).max(-1)
    res["by_class"] = {}
    for nm, msk in (("floor", floor), ("wall_top", wtop), ("wall_side", wside), ("agent", agm)):
        mk = msk & ~b
        if mk.sum() == 0:
            continue
        res["by_class"][nm] = dict(n=int(mk.sum()), exact=round(float((dd[mk] == 0).mean()), 5),
                                   le1=round(float((dd[mk] <= 1).mean()), 5),
                                   mean=round(float(dd[mk].mean()), 4), max=float(dd[mk].max()))

    # ---- Gouraud test: is each face's colour bilinear in the face's own (u,v)? ----
    gt_res, pp_res, ns = [], [], []
    tops = [g for g in np.unique(gid[wtop]) if g > 0]
    for g in tops[:24]:
        mk = wtop & (gid == g) & ~b & ~sat
        if mk.sum() < 60:
            continue
        pos = d.geom_xpos[g]
        uu = p[mk][:, 0] - pos[0]
        vv = p[mk][:, 1] - pos[1]
        y = obs[mk].mean(-1)
        A = np.stack([np.ones_like(uu), uu, vv, uu * vv], 1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        gt_res.append(float(np.abs(A @ c - y).max()))
        pp_res.append(float(np.abs(pred[mk].mean(-1) - y).max()))
        ns.append(int(mk.sum()))
    res["wall_top_faces"] = dict(n_faces=len(gt_res),
                                 bilinear_max_resid=round(float(np.mean(gt_res)), 4) if gt_res else None,
                                 perpixel_max_resid=round(float(np.mean(pp_res)), 4) if pp_res else None,
                                 mean_face_px=int(np.mean(ns)) if ns else 0)
    # same test on the floor, for contrast (the floor should NOT be a single bilinear patch)
    mk = floor & ~b & ~sat
    uu = p[mk][:, 0]
    vv = p[mk][:, 1]
    y = obs[mk].mean(-1)
    A = np.stack([np.ones_like(uu), uu, vv, uu * vv], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res["floor_bilinear_max_resid"] = round(float(np.abs(A @ c - y).max()), 4)

    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    allr = []
    log = open(os.path.join(OUT, "fit2.log"), "w")
    for loco, size in [("point", "medium"), ("point", "large"), ("humanoid", "large")]:
        try:
            r = run(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {loco}-{size}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        allr.append(r)
        log.write(json.dumps(r, indent=1) + "\n")
        log.flush()
        print(json.dumps(r, indent=1), flush=True)
    log.close()
    with open(os.path.join(OUT, "fit2.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
