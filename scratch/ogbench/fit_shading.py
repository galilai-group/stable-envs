"""Fit MuJoCo's shading scalar instead of deriving it from the light table.

``analytic_rgb.py`` used the textbook reading of the model -- ambient .1, diffuse 1.4, Blinn specular
.3*(n.h)^64 -- and landed 18% exact / 2.5 mean, i.e. right to a couple of levels and wrong in detail.
Rather than guess which detail (local vs infinite viewer, Blinn vs Phong, where the specular is folded
in), the scalar is recovered from the pixels: on the floor the normal is known and constant, the base
colour is known, so ``S = obs/base`` is observable directly, and it can be regressed against both
candidate forms over a grid of exponents. Whichever form fits to <1/255 is the one MuJoCo uses.

Then the fitted S is checked where it was not fitted: wall tops (same normal, different z and different
base colour) and the agent (curved normals), which is the only real test of the functional form.
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

    res = dict(env=f"{loco}maze-{size}")

    # ---- observable S on the floor ----------------------------------------
    Sf = obs[floor] / (base[floor] * 255.0)
    Sf = Sf.mean(1)                                     # channels agree; average kills quantisation noise
    ndlf = np.clip((n[floor] @ L), 0, None)
    # candidate 1: Blinn, LOCAL viewer  h = normalize(l + v)
    hv = L + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)
    ndh_local = np.clip((n * hv).sum(-1), 0, None)
    # candidate 2: Blinn, INFINITE viewer  h = normalize(l + z_eye); z_eye = -camera forward
    zeye = -np.asarray(cam["forward"], np.float64)
    zeye = zeye / np.linalg.norm(zeye)
    h_inf = L + zeye
    h_inf = h_inf / np.linalg.norm(h_inf)
    ndh_inf = np.clip(n @ h_inf, 0, None)
    # candidate 3: Phong  r = 2(n.l)n - l
    rv = 2 * (n @ L)[..., None] * n - L
    rdv = np.clip((rv * V).sum(-1), 0, None)

    fits = {}
    for nm, x in (("blinn_local", ndh_local), ("blinn_infinite", ndh_inf), ("phong", rdv)):
        xf = x[floor]
        best = None
        for pexp in list(range(1, 129)) + [0.5]:
            A = np.stack([np.ones_like(xf), ndlf, xf ** pexp], 1)
            coef, *_ = np.linalg.lstsq(A, Sf, rcond=None)
            resid = A @ coef - Sf
            rms = float(np.sqrt((resid ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, pexp, coef.tolist())
        fits[nm] = dict(rms_S=best[0], rms_255=best[0] * 60.0, exponent=best[1],
                        coef_amb_diff_spec=[round(c, 6) for c in best[2]])
    res["floor_fits"] = fits
    best_form = min(fits, key=lambda k: fits[k]["rms_S"])
    res["best_form"] = best_form
    xmap = {"blinn_local": ndh_local, "blinn_infinite": ndh_inf, "phong": rdv}[best_form]
    pexp = fits[best_form]["exponent"]
    ca, cd, cs = fits[best_form]["coef_amb_diff_spec"]
    res["fitted"] = dict(ambient=ca, diffuse=cd, specular=cs, exponent=pexp, form=best_form)

    # ---- apply the fit everywhere and score --------------------------------
    S = ca + cd * np.clip((n @ L), 0, None) + cs * xmap ** pexp
    pred = np.clip(base * S[..., None], 0, 1) * 255.0
    pred[gid < 0] = obs[gid < 0]                    # skybox is baked, not modelled here
    pr = np.clip(np.round(pred), 0, 255)
    dd = np.abs(pr - obs).max(-1)
    b = np.zeros(gid.shape, bool)
    b[1:, :] |= gid[1:, :] != gid[:-1, :]
    b[:-1, :] |= gid[1:, :] != gid[:-1, :]
    b[:, 1:] |= gid[:, 1:] != gid[:, :-1]
    b[:, :-1] |= gid[:, 1:] != gid[:, :-1]
    res["fitted_score"] = dict(exact=float((dd == 0).mean()), le1=float((dd <= 1).mean()),
                               mean=float(dd.mean()), max=float(dd.max()),
                               exact_interior=float((dd[~b] == 0).mean()),
                               le1_interior=float((dd[~b] <= 1).mean()),
                               max_interior=float(dd[~b].max()))
    res["fitted_by_class"] = {}
    for nm, msk in (("floor", floor), ("wall_top", wtop), ("wall_side", wside), ("agent", agm)):
        mk = msk & ~b
        if mk.sum() == 0:
            continue
        res["fitted_by_class"][nm] = dict(n=int(mk.sum()), exact=float((dd[mk] == 0).mean()),
                                          le1=float((dd[mk] <= 1).mean()), mean=float(dd[mk].mean()),
                                          max=float(dd[mk].max()))

    # residual structure on the floor: is it random quantisation or a spatial pattern?
    rf = (pr - obs).max(-1)[floor & ~b]
    res["floor_residual_hist"] = {str(int(k)): int(v) for k, v in
                                  zip(*np.unique(np.round(rf), return_counts=True))}
    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    allr = []
    for loco, size in [("point", "medium"), ("point", "large"), ("ant", "large"),
                       ("humanoid", "large")]:
        try:
            r = run(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        allr.append(r)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "fit_shading.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
