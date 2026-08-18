"""The correct composition: specular is ADDITIVE and white, and the texture modulates the whole thing.

The floor said the shading was a pure multiplier (channel spread of ``obs/base`` = 3e-4). The agent says
it is not (channel spread 6.8e-2). Both are true, and one model explains both -- fixed-function OpenGL:

    primary = mat_rgb * (A + D*max(0, n.l))  +  Sp * max(0, n.h)^p        (specular is WHITE, added)
    final   = primary * tex_rgb                                            (tex = 1 when untextured)

The floor's material is ``grid``, whose ``mat_rgba`` is (1,1,1) and whose colour comes entirely from a
texture, so ``final = tex * (A + D + Sp*(n.h)^p)`` -- a pure multiplier, which is what was measured.
Walls and the agent have no texture, so their specular is a white additive term, which is why a
multiplicative law fitted on the floor overshoots wall tops by 6.5/255 and mangles the agent's highlight.

Fitted jointly over floor + wall tops + wall sides + agent, all channels, over a grid of exponents.
Scored on AGENT PIXELS ONLY, which is the number that matters.
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

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
L = np.array([0.0, 0.0, 1.0])


def mat_and_tex(m, g):
    """(mat_rgb, tex_rgb) for geom ``g`` under the composition above."""
    mid = int(m.geom_matid[g])
    mat = m.geom_rgba[g].astype(np.float64)[:3]
    tex = np.ones(3)
    if mid >= 0:
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            mat = m.mat_rgba[mid].astype(np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = buf[a0:a0 + 3 * h * w].reshape(h, w, 3).astype(np.float64) / 255.0
            uniq = np.unique(reg.reshape(-1, 3), axis=0)
            tex = uniq[0] if len(uniq) == 1 else reg.reshape(-1, 3).mean(0)
    return mat, tex


def collect(loco, size, img=224, nstates=40, marker=1.0):
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
    ag, names = agent_geoms(m)
    static = [g for g in range(m.ngeom) if g not in ag]
    sg, bg_t, o, dr = cast_subset(m, d, cam, img, img, static)
    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    hv = L + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)

    S = {k: [] for k in ("ndl", "ndh", "mat", "tex", "obs", "cls")}

    def push(msk, gid_map, n, obs, cls):
        if not msk.any():
            return
        mat = np.zeros((img, img, 3))
        tex = np.ones((img, img, 3))
        for g in np.unique(gid_map[msk]):
            mm, tt = mat_and_tex(m, int(g))
            sel = msk & (gid_map == g)
            mat[sel] = mm
            tex[sel] = tt
        S["ndl"].append(np.clip(n @ L, 0, None)[msk])
        S["ndh"].append(np.clip((n * hv).sum(-1), 0, None)[msk])
        S["mat"].append(mat[msk])
        S["tex"].append(tex[msk])
        S["obs"].append(obs[msk])
        S["cls"].append(np.full(int(msk.sum()), cls))

    for _ in range(nstates):
        for _ in range(9):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        gid, t, _, _ = cast_subset(m, d, cam, img, img, ag)
        vis = (t < INF / 2) & (t < bg_t)
        p = o + t[..., None] * dr
        if vis.any():
            n = normals(m, d, np.where(vis, gid, -1), p)
            e = vis.copy()
            e[1:] &= vis[:-1]
            e[:-1] &= vis[1:]
            e[:, 1:] &= vis[:, :-1]
            e[:, :-1] &= vis[:, 1:]
            push(e, gid, n, obs, 3)
        break_static = len(S["obs"]) > 0
        if break_static:                                 # static classes once: they do not move
            ps = o + bg_t[..., None] * dr
            ns = normals(m, d, sg, ps)
            wall = np.isin(sg, [g for g in range(m.ngeom)
                                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                                .startswith("block_")])
            bnd = np.zeros(sg.shape, bool)
            bnd[1:] |= sg[1:] != sg[:-1]
            bnd[:-1] |= sg[1:] != sg[:-1]
            bnd[:, 1:] |= sg[:, 1:] != sg[:, :-1]
            bnd[:, :-1] |= sg[:, 1:] != sg[:, :-1]
            free = ~vis & ~bnd
            push((sg == 0) & free, sg, ns, obs, 0)
            push(wall & (ns[..., 2] > 0.5) & free & (obs < 254.5).all(-1), sg, ns, obs, 1)
            push(wall & (ns[..., 2] <= 0.5) & free, sg, ns, obs, 2)
            break
    # agent pixels from the remaining states
    for _ in range(nstates):
        for _ in range(9):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        gid, t, _, _ = cast_subset(m, d, cam, img, img, ag)
        vis = (t < INF / 2) & (t < bg_t)
        if not vis.any():
            continue
        p = o + t[..., None] * dr
        n = normals(m, d, np.where(vis, gid, -1), p)
        e = vis.copy()
        e[1:] &= vis[:-1]
        e[:-1] &= vis[1:]
        e[:, 1:] &= vis[:, :-1]
        e[:, :-1] &= vis[:, 1:]
        push(e, gid, n, obs, 3)
    env.close()
    return {k: np.concatenate(v) for k, v in S.items()}


def design(ndl, ndh, mat, tex, p):
    """Rows for `obs/255 = tex*(mat*(A + D*ndl) + Sp*ndh^p)`, flattened over channels."""
    a = (tex * mat).reshape(-1)
    b = (tex * mat * ndl[:, None]).reshape(-1)
    c = (tex * (ndh ** p)[:, None]).reshape(-1)
    return np.stack([a, b, c], 1)


def predict(ndl, ndh, mat, tex, coef, p):
    A, D, Sp = coef
    prim = mat * (A + D * ndl[:, None]) + Sp * (ndh ** p)[:, None]
    return np.clip(np.round(np.clip(prim * tex, 0, 1) * 255.0), 0, 255)


def run(loco, size, marker=1.0):
    S = collect(loco, size, marker=marker)
    ndl, ndh, mat, tex, obs, cls = (S[k] for k in ("ndl", "ndh", "mat", "tex", "obs", "cls"))
    unsat = (obs < 254.5).all(-1)
    best = None
    for p in range(1, 129):
        Xd = design(ndl[unsat], ndh[unsat], mat[unsat], tex[unsat], p)
        y = (obs[unsat] / 255.0).reshape(-1)
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        rms = float(np.sqrt(((Xd @ coef - y) ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, p, coef)
    rms, p, coef = best
    out = dict(env=f"{loco}maze-{size}", marker=marker,
               fit=dict(ambient=round(float(coef[0]), 6), diffuse=round(float(coef[1]), 6),
                        specular=round(float(coef[2]), 6), exponent=int(p), rms=round(rms, 7)),
               n=dict(floor=int((cls == 0).sum()), wall_top=int((cls == 1).sum()),
                      wall_side=int((cls == 2).sum()), agent=int((cls == 3).sum())))
    pred = predict(ndl, ndh, mat, tex, coef, p)
    dd = np.abs(pred - obs).max(-1)
    for nm, c in (("floor", 0), ("wall_top", 1), ("wall_side", 2), ("agent", 3)):
        k = cls == c
        if k.sum() == 0:
            continue
        out[nm] = dict(n=int(k.sum()), exact=round(float((dd[k] == 0).mean()), 5),
                       le1=round(float((dd[k] <= 1).mean()), 5),
                       le2=round(float((dd[k] <= 2).mean()), 5),
                       mean=round(float(dd[k].mean()), 4), max=float(dd[k].max()))
    # round-half-up vs round-half-even: OpenGL's float->unorm8 is round-to-nearest
    prim = mat * (coef[0] + coef[1] * ndl[:, None]) + coef[2] * (ndh ** p)[:, None]
    v = np.clip(prim * tex, 0, 1) * 255.0
    for nm, f in (("rint", np.rint), ("floor+.5", lambda x: np.floor(x + 0.5))):
        d2 = np.abs(np.clip(f(v), 0, 255) - obs).max(-1)
        k = cls == 3
        out[f"agent_exact_{nm}"] = round(float((d2[k] == 0).mean()), 5)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "fit3.log"), "w")
    allr = []
    for loco, size, mk in [("point", "medium", 1.0), ("point", "medium", 3.0),
                           ("point", "large", 1.0), ("ant", "large", 3.0),
                           ("humanoid", "large", 4.0)]:
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
        print(json.dumps(r), flush=True)
    log.close()
    with open(os.path.join(OUT, "fit3.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
