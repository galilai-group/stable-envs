"""Stop guessing the specular form: recover it nonparametrically, then read off what it is.

Where this stands. The composition is settled -- specular is white and ADDITIVE on untextured surfaces,
and the texture modulates the whole primary colour; that took wall tops from 13/255 to <=2/255. The
tessellation hypothesis is dead: every (slices, stacks, phase, diagonal) variant of a Gouraud sphere
scored 0.0-0.2% exact on agent pixels against 1.6% for plain per-pixel, so MuJoCo is not interpolating
vertex shades here. What is left is that one Blinn lobe cannot serve the floor (wants exponent ~63) and
the agent (wants ~46) at once, which means the FORM is wrong, not the fit.

So the lobe is not assumed at all. With the base colour and the normal known per pixel, the specular
contribution is observable:

    s = obs/255 / tex  -  mat * (A + sum_l D_l * max(0, n.l_l))

and it must be identical across R, G and B (it is white). Fitting ``A``, the per-light diffuse weights
and a piecewise-linear lobe in the candidate argument recovers the curve without naming it. Two light
geometries are scored, because MuJoCo's headlight reports ``type = directional`` but sits at the camera:

    (i)  headlight directional along the camera axis  -- what the light table says
    (ii) headlight as a POINT light at the camera     -- l varies per pixel

and two specular arguments, Blinn ``n.h`` and Phong ``r.v``. Whichever pair explains floor AND agent with
one curve is the renderer's actual model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from general_check import make2, agent_geoms, cast_subset  # noqa: E402
from fit3 import mat_and_tex  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
K = 24                       # hat-basis knots for the nonparametric lobe


def hats(x, K=K):
    """(N,K) piecewise-linear basis on [0,1]."""
    t = np.clip(x, 0, 1) * (K - 1)
    i = np.clip(np.floor(t).astype(int), 0, K - 2)
    f = t - i
    B = np.zeros((len(x), K))
    B[np.arange(len(x)), i] = 1 - f
    B[np.arange(len(x)), i + 1] = f
    return B


def gather(loco, size, img=224, nstates=30, marker=3.0):
    import mujoco
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    if marker != 1.0:
        ag0, _ = agent_geoms(m)
        for g in ag0:
            m.geom_size[g] = m.geom_size[g] * marker
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    campos = np.asarray(cam["pos"], np.float64)
    camfwd = np.asarray(cam["forward"], np.float64)
    ag, _ = agent_geoms(m)
    static = [g for g in range(m.ngeom) if g not in ag]
    sg, bg_t, o3, dr3 = cast_subset(m, d, cam, img, img, static)

    rows = {k: [] for k in ("n", "p", "mat", "tex", "obs", "cls")}

    def push(msk, gmap, nrm, pos, obs, cls):
        if not msk.any():
            return
        mat = np.zeros((img, img, 3))
        tex = np.ones((img, img, 3))
        for g in np.unique(gmap[msk]):
            mm, tt = mat_and_tex(m, int(g))
            sel = msk & (gmap == g)
            mat[sel] = mm
            tex[sel] = tt
        rows["n"].append(nrm[msk])
        rows["p"].append(pos[msk])
        rows["mat"].append(mat[msk])
        rows["tex"].append(tex[msk])
        rows["obs"].append(obs[msk])
        rows["cls"].append(np.full(int(msk.sum()), cls))

    got_static = False
    for _ in range(nstates):
        for _ in range(9):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        gid, t, _, _ = cast_subset(m, d, cam, img, img, ag)
        vis = (t < INF / 2) & (t < bg_t)
        if vis.any():
            p = o3 + t[..., None] * dr3
            n = normals(m, d, np.where(vis, gid, -1), p)
            e = vis.copy()
            e[1:] &= vis[:-1]
            e[:-1] &= vis[1:]
            e[:, 1:] &= vis[:, :-1]
            e[:, :-1] &= vis[:, 1:]
            push(e & (obs < 254.5).all(-1), gid, n, p, obs, 2)
        if not got_static:
            ps = o3 + bg_t[..., None] * dr3
            ns = normals(m, d, sg, ps)
            bnd = np.zeros(sg.shape, bool)
            bnd[1:] |= sg[1:] != sg[:-1]
            bnd[:-1] |= sg[1:] != sg[:-1]
            bnd[:, 1:] |= sg[:, 1:] != sg[:, :-1]
            bnd[:, :-1] |= sg[:, 1:] != sg[:, :-1]
            free = ~bnd & ~vis & (obs < 254.5).all(-1) & (sg >= 0)
            push(free & (sg == 0), sg, ns, ps, obs, 0)
            push(free & (sg != 0), sg, ns, ps, obs, 1)
            got_static = True
    env.close()
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["campos"], out["camfwd"] = campos, camfwd
    return out


def variant(S, headlight_point: bool, phong: bool):
    n, p, mat, tex, obs = S["n"], S["p"], S["mat"], S["tex"], S["obs"]
    V = S["campos"] - p
    V = V / np.linalg.norm(V, axis=-1, keepdims=True)
    lg = np.broadcast_to(np.array([0.0, 0.0, 1.0]), n.shape)          # global light, directional
    lh = V if headlight_point else np.broadcast_to(-S["camfwd"], n.shape)
    ndl_g = np.clip((n * lg).sum(-1), 0, None)
    ndl_h = np.clip((n * lh).sum(-1), 0, None)
    if phong:
        rg = 2 * (n * lg).sum(-1)[:, None] * n - lg
        rh = 2 * (n * lh).sum(-1)[:, None] * n - lh
        xg = np.clip((rg * V).sum(-1), 0, None)
        xh = np.clip((rh * V).sum(-1), 0, None)
    else:
        hg = lg + V
        hg = hg / np.linalg.norm(hg, axis=-1, keepdims=True)
        hh = lh + V
        hh = hh / np.linalg.norm(hh, axis=-1, keepdims=True)
        xg = np.clip((n * hg).sum(-1), 0, None)
        xh = np.clip((n * hh).sum(-1), 0, None)
    # both lights share a direction when the camera looks straight down, so one lobe argument suffices;
    # use the brighter light's (headlight, specular .5) and let the fit absorb the rest
    x = np.maximum(xg, xh)
    cols = [(tex * mat).reshape(-1, 1),
            (tex * mat * ndl_g[:, None]).reshape(-1, 1),
            (tex * mat * ndl_h[:, None]).reshape(-1, 1)]
    Bx = hats(x)
    cols.append((tex[:, :, None] * Bx[:, None, :]).reshape(-1, K))
    X = np.concatenate(cols, 1)
    y = (obs / 255.0).reshape(-1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.clip(np.round(np.clip((X @ coef).reshape(-1, 3), 0, 1) * 255), 0, 255)
    dd = np.abs(pred - obs).max(-1)
    cls = S["cls"]
    out = dict(rms=float(np.sqrt(((X @ coef - y) ** 2).mean())),
               amb=round(float(coef[0]), 5), d_global=round(float(coef[1]), 5),
               d_head=round(float(coef[2]), 5),
               lobe=[round(float(c), 5) for c in coef[3:]])
    for nm, c in (("floor", 0), ("static_other", 1), ("agent", 2)):
        k = cls == c
        if k.sum():
            out[nm] = dict(n=int(k.sum()), exact=round(float((dd[k] == 0).mean()), 5),
                           le1=round(float((dd[k] <= 1).mean()), 5),
                           mean=round(float(dd[k].mean()), 4), max=float(dd[k].max()))
    return out


def run(loco, size, marker):
    S = gather(loco, size, marker=marker)
    res = dict(env=f"{loco}maze-{size}", marker=marker,
               n=dict(floor=int((S["cls"] == 0).sum()), static_other=int((S["cls"] == 1).sum()),
                      agent=int((S["cls"] == 2).sum())))
    for hp in (False, True):
        for ph in (False, True):
            res[f"head_point={hp}_phong={ph}"] = variant(S, hp, ph)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "fit5.log"), "w")
    allr = []
    for loco, size, mk in [("point", "medium", 3.0), ("humanoid", "large", 4.0)]:
        try:
            r = run(loco, size, mk)
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
    with open(os.path.join(OUT, "fit5.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
