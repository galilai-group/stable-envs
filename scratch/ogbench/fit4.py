"""Gouraud, tested: ray-cast MuJoCo's TESSELLATION and interpolate vertex shades, on agent pixels only.

With the composition corrected (specular additive and white; texture modulates the whole primary colour)
wall tops fall to <=2/255 and the floor to <=1/255, but a single specular exponent cannot serve both the
floor (which wants ~63) and the agent (which wants ~46). That split is the Gouraud signature: MuJoCo
lights per vertex, a sphere of radius 0.7 at this camera is ~125 px covered by ~224 front-facing quads,
and the vertex spacing (360/28 = 12.9 deg of longitude, 180/16 = 11.25 deg of latitude) is comparable to
the half-width of a ``(n.h)^60`` lobe. Sampling a peaked lobe that coarsely and interpolating linearly
FLATTENS it, which is exactly what a lower fitted exponent means.

So: build the triangle mesh, intersect it, shade the three vertices, interpolate barycentrically. The
tessellation convention is not read off the MuJoCo source, it is scored -- pole offset, latitude
convention and quad diagonal are all enumerated and the variant that reproduces the pixels wins.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from general_check import make2, agent_geoms, cast_subset  # noqa: E402
from fit3 import mat_and_tex, design  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
L = np.array([0.0, 0.0, 1.0])


def sphere_mesh(nslice, nstack, phi0=0.0, flip=False):
    """Unit sphere vertices (V,3) and triangles (T,3) for one tessellation hypothesis."""
    th = np.linspace(0, np.pi, nstack + 1)
    if flip:
        th = th[::-1]
    ph = np.arange(nslice) * (2 * np.pi / nslice) + phi0
    V, idx = [], {}
    for j, t in enumerate(th):
        for i in range(nslice):
            idx[(i, j)] = len(V)
            V.append([np.sin(t) * np.cos(ph[i]), np.sin(t) * np.sin(ph[i]), np.cos(t)])
    V = np.asarray(V)
    return V, idx


def quads_to_tris(idx, nslice, nstack, diag):
    T = []
    for j in range(nstack):
        for i in range(nslice):
            a = idx[(i, j)]
            b = idx[((i + 1) % nslice, j)]
            c = idx[((i + 1) % nslice, j + 1)]
            d = idx[(i, j + 1)]
            if diag == 0:
                T += [[a, b, c], [a, c, d]]
            else:
                T += [[a, b, d], [b, c, d]]
    return np.asarray(T, np.int64)


def moller(o, dr, v0, v1, v2):
    """(P,T) t and barycentrics for P rays against T triangles. NaN where no hit."""
    e1 = v1 - v0
    e2 = v2 - v0
    pv = np.cross(dr[:, None, :], e2[None, :, :])
    det = (e1[None] * pv).sum(-1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o[:, None, :] - v0[None]
    u = (tv * pv).sum(-1) * inv
    qv = np.cross(tv, e1[None])
    v = (dr[:, None, :] * qv).sum(-1) * inv
    t = (e2[None] * qv).sum(-1) * inv
    good = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
    t = np.where(good, t, np.inf)
    return t, u, v


def run(loco, size, img=224, nstates=30, marker=1.0):
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
    sg, bg_t, o3, dr3 = cast_subset(m, d, cam, img, img, static)
    V = -dr3 / np.linalg.norm(dr3, axis=-1, keepdims=True)
    hv = L + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)
    res = dict(env=f"{loco}maze-{size}", marker=marker,
               quality=dict(numslices=int(m.vis.quality.numslices),
                            numstacks=int(m.vis.quality.numstacks),
                            numquads=int(m.vis.quality.numquads)))

    # ---- 1. pin A, D, Sp, p on STATIC pixels only (floor + walls), additive model -------
    r.update_scene(d, camera=u.custom_camera)
    obs0 = r.render().astype(np.float64)
    ns = normals(m, d, sg, o3 + bg_t[..., None] * dr3)
    bnd = np.zeros(sg.shape, bool)
    bnd[1:] |= sg[1:] != sg[:-1]
    bnd[:-1] |= sg[1:] != sg[:-1]
    bnd[:, 1:] |= sg[:, 1:] != sg[:, :-1]
    bnd[:, :-1] |= sg[:, 1:] != sg[:, :-1]
    gidA, tA, _, _ = cast_subset(m, d, cam, img, img, ag)
    stat = ~bnd & ~((tA < INF / 2) & (tA < bg_t)) & (obs0 < 254.5).all(-1) & (sg >= 0)
    mat = np.zeros((img, img, 3))
    tex = np.ones((img, img, 3))
    for g in np.unique(sg[stat]):
        mm, tt = mat_and_tex(m, int(g))
        sel = stat & (sg == g)
        mat[sel] = mm
        tex[sel] = tt
    ndl_s = np.clip(ns @ L, 0, None)[stat]
    ndh_s = np.clip((ns * hv).sum(-1), 0, None)[stat]
    y = (obs0[stat] / 255.0).reshape(-1)
    best = None
    for p in range(1, 129):
        X = design(ndl_s, ndh_s, mat[stat], tex[stat], p)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rms = float(np.sqrt(((X @ coef - y) ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, p, coef)
    rms, P, coef = best
    A, D, Sp = coef
    res["static_fit"] = dict(ambient=round(float(A), 6), diffuse=round(float(D), 6),
                             specular=round(float(Sp), 6), exponent=int(P), rms=round(rms, 7),
                             n=int(stat.sum()))
    pred = np.clip(np.round(np.clip((mat[stat] * (A + D * ndl_s[:, None])
                                     + Sp * (ndh_s ** P)[:, None]) * tex[stat], 0, 1) * 255), 0, 255)
    dd = np.abs(pred - obs0[stat]).max(-1)
    res["static_score"] = dict(exact=round(float((dd == 0).mean()), 5),
                               le1=round(float((dd <= 1).mean()), 5), max=float(dd.max()))

    def shade(nrm, pos):
        """Additive-specular shading at world points ``pos`` with normals ``nrm`` (both (...,3))."""
        vv = cam_pos - pos
        vv = vv / np.linalg.norm(vv, axis=-1, keepdims=True)
        h = L + vv
        h = h / np.linalg.norm(h, axis=-1, keepdims=True)
        return (np.clip((nrm * L).sum(-1), 0, None), np.clip((nrm * h).sum(-1), 0, None))

    cam_pos = np.asarray(cam["pos"], np.float64)

    # ---- 2. agent pixels: per-pixel vs Gouraud over a grid of tessellation hypotheses ----
    sphere_ids = [g for g in ag if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_SPHERE]
    if not sphere_ids:
        env.close()
        return res
    gsph = sphere_ids[0]
    matS, texS = mat_and_tex(m, gsph)
    NSL, NST = int(m.vis.quality.numslices), int(m.vis.quality.numstacks)
    hyps = [(nsl, nst, phi0, flip, diag)
            for nsl in (NSL,) for nst in (NST,)
            for phi0 in (0.0, np.pi / NSL) for flip in (False, True) for diag in (0, 1)]
    hyps += [(NSL, NST // 2, 0.0, False, 0), (NSL * 2, NST, 0.0, False, 0)]
    tally = {str(h): [0, 0] for h in hyps}
    pp = [0, 0]
    rng = np.random.default_rng(0)
    for _ in range(nstates):
        for _ in range(9):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        gid, t, _, _ = cast_subset(m, d, cam, img, img, [gsph])
        vis = (t < INF / 2) & (t < bg_t)
        e = vis.copy()
        e[1:] &= vis[:-1]
        e[:-1] &= vis[1:]
        e[:, 1:] &= vis[:, :-1]
        e[:, :-1] &= vis[:, 1:]
        if e.sum() < 5:
            continue
        ij = np.argwhere(e)
        rays_o = o3[e]
        rays_d = dr3[e]
        target = obs[e]
        # per-pixel (analytic surface)
        ph = rays_o + t[e][:, None] * rays_d
        nn = (ph - d.geom_xpos[gsph]) / m.geom_size[gsph][0]
        ndl, ndh = shade(nn, ph)
        pr = np.clip(np.round(np.clip((matS * (A + D * ndl[:, None]) + Sp * (ndh ** P)[:, None])
                                      * texS, 0, 1) * 255), 0, 255)
        pp[0] += int((np.abs(pr - target).max(-1) == 0).sum())
        pp[1] += len(target)
        # Gouraud over each hypothesis
        c0 = d.geom_xpos[gsph]
        rad = float(m.geom_size[gsph][0])
        for h in hyps:
            nsl, nst, phi0, flip, diag = h
            Vu, idx = sphere_mesh(nsl, nst, phi0, flip)
            T = quads_to_tris(idx, nsl, nst, diag)
            Vw = c0 + rad * Vu
            ndlv, ndhv = shade(Vu, Vw)
            Sv = matS * (A + D * ndlv[:, None]) + Sp * (ndhv ** P)[:, None]
            tt, uu, vv = moller(rays_o, rays_d, Vw[T[:, 0]], Vw[T[:, 1]], Vw[T[:, 2]])
            k = np.argmin(tt, 1)
            hitv = np.isfinite(tt[np.arange(len(k)), k])
            w1 = uu[np.arange(len(k)), k]
            w2 = vv[np.arange(len(k)), k]
            w0 = 1 - w1 - w2
            col = (w0[:, None] * Sv[T[k, 0]] + w1[:, None] * Sv[T[k, 1]] + w2[:, None] * Sv[T[k, 2]])
            pg = np.clip(np.round(np.clip(col * texS, 0, 1) * 255), 0, 255)
            ok = hitv
            tally[str(h)][0] += int((np.abs(pg[ok] - target[ok]).max(-1) == 0).sum())
            tally[str(h)][1] += int(ok.sum())
    res["agent_perpixel_exact"] = round(pp[0] / max(pp[1], 1), 5)
    res["agent_n"] = pp[1]
    res["gouraud"] = {k: dict(exact=round(v[0] / max(v[1], 1), 5), n=v[1]) for k, v in tally.items()}
    res["gouraud_best"] = max(res["gouraud"], key=lambda k: res["gouraud"][k]["exact"])
    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "fit4.log"), "w")
    allr = []
    for loco, size, mk in [("point", "medium", 1.0), ("point", "medium", 3.0)]:
        try:
            r = run(loco, size, marker=mk)
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
    with open(os.path.join(OUT, "fit4.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
