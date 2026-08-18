"""Fit the shading law on AGENT pixels, through the packaged renderer's own geometry.

The plate carries every static pixel exactly, so the analytic shader is only ever asked about the
agent. Calibrating it on floor pixels — where ``n.l`` is 1 everywhere and ``n.h`` never leaves a few
degrees of 1 — is therefore fitting on the one region the shader never has to serve. This refits
(ambient, diffuse, specular, exponent) on agent-interior pixels, using the package's ``Camera`` and
intersectors so the numbers apply to exactly the code path that will ship, and sweeps the supersample
factor because the reference is MSAA'd and the agent is mostly silhouette.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"


def agent_pixels(e, states, n=None):
    """(mask, normals, view dirs, obs) for agent-interior pixels, from MuJoCo's segmentation."""
    import mujoco
    from stable_envs.render import raycast3d as R3
    cls = type(e)
    ref = cls.reference_env(size=e.size)
    u = ref.unwrapped
    m = u.model
    sc = e._scene()
    o, d = sc.cam.rays(e.size, e.size, e.device)
    ids = sc.agent_ids
    prims = R3.Primitives(
        kind=np.array([int(sc.model.geom_type[g]) for g in ids]),
        size=np.stack([sc.model.geom_size[g] for g in ids]),
        mat_rgb=np.ones((len(ids), 3)), tex_rgb=np.ones((len(ids), 3)),
        spec=np.full(len(ids), 0.5))
    P = prims.to(e.device)
    rows = {k: [] for k in ("ndl", "ndh", "obs")}
    lam = sc.lights
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=e.size, height=e.size)
    for st in np.asarray(states, np.float64):
        u.set_state(st, np.zeros(m.nv))
        r.enable_segmentation_rendering()
        r.update_scene(u.data, camera=u.custom_camera)
        seg = r.render()
        r.disable_segmentation_rendering()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        am = np.isin(gid, ids)
        ai = am.copy()
        ai[1:] &= am[:-1]
        ai[:-1] &= am[1:]
        ai[:, 1:] &= am[:, :-1]
        ai[:, :-1] &= am[:, 1:]
        if ai.sum() < 3:
            continue
        r.update_scene(u.data, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        xp, xm = e._fk(torch.as_tensor(st, device=e.device, dtype=torch.float32).view(1, -1), sc)
        N = e.size * e.size
        best = torch.full((N,), R3.INF, device=e.device)
        nrm = torch.zeros(N, 3, device=e.device)
        for i, k in enumerate(P["kind"]):
            t, nn = R3.intersect(k, P["size"][i].view(1, 3).expand(N, 3),
                                 xp[0, i].view(1, 3).expand(N, 3),
                                 xm[0, i].view(1, 3, 3).expand(N, 3, 3), o.expand(N, 3), d)
            upd = t < best
            best = torch.where(upd, t, best)
            nrm = torch.where(upd.unsqueeze(-1), nn, nrm)
        v = (-d).cpu().numpy()
        nn = nrm.cpu().numpy()
        flat = ai.reshape(-1)
        hit = (best < R3.INF / 2).cpu().numpy() & flat
        if hit.sum() == 0:
            continue
        L = lam.dirs
        ndl = np.stack([np.clip(nn[hit] @ L[i], 0, None) for i in range(len(L))], 1)
        ndh = []
        for i in range(len(L)):
            h = L[i] + v[hit]
            h = h / np.linalg.norm(h, axis=-1, keepdims=True)
            ndh.append(np.clip((nn[hit] * h).sum(-1), 0, None))
        rows["ndl"].append(ndl)
        rows["ndh"].append(np.stack(ndh, 1))
        rows["obs"].append(obs.reshape(-1, 3)[hit])
    m.vis.quality.offsamples = 4
    return {k: np.concatenate(v) for k, v in rows.items() if v}


def fit(rows, mat, wd, ws):
    """Least squares for (A, D, Sp) over a grid of exponents, with the light split held fixed."""
    ndl, ndh, obs = rows["ndl"], rows["ndh"], rows["obs"]
    unsat = (obs < 254.5).all(-1)
    ndl, ndh, obs = ndl[unsat], ndh[unsat], obs[unsat]
    dif = (ndl * wd).sum(1) / max(wd.sum(), 1e-9)
    best = None
    for p in range(1, 129):
        spec = ((ndh ** p) * ws).sum(1) / max(ws.sum(), 1e-9)
        X = np.concatenate([np.tile(mat, (len(dif), 1)).reshape(-1, 1),
                            (np.tile(mat, (len(dif), 1)) * dif[:, None]).reshape(-1, 1),
                            np.tile(spec[:, None], (1, 3)).reshape(-1, 1)], 1)
        y = (obs / 255.0).reshape(-1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rms = float(np.sqrt(((X @ coef - y) ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, p, coef)
    rms, p, coef = best
    spec = ((ndh ** p) * ws).sum(1) / max(ws.sum(), 1e-9)
    pred = np.clip(np.round(np.clip(mat * (coef[0] + coef[1] * dif[:, None])
                                    + coef[2] * spec[:, None], 0, 1) * 255), 0, 255)
    dd = np.abs(pred - obs).max(-1)
    return dict(ambient=round(float(coef[0]), 6), diffuse=round(float(coef[1]), 6),
                specular=round(float(coef[2]), 6), exponent=int(p), rms=round(rms, 7),
                n=int(len(dd)), exact=round(float((dd == 0).mean()), 5),
                le1=round(float((dd <= 1).mean()), 5), le2=round(float((dd <= 2).mean()), 5),
                mean=round(float(dd.mean()), 4), max=float(dd.max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    import stable_envs as se
    log = open(os.path.join(OUT, "calibrate_agent.log"), "w")
    res = []
    # Pooled over the four pointmaze scenes. Only pointmaze has agent-INTERIOR pixels at the official
    # camera at all — the ant is ~6 px and the humanoid ~7, so their agents are pure silhouette and
    # carry no information about the shading law. The law is a property of the renderer, not of the
    # body, so calibrating it on the one body whose normals sweep a full hemisphere and applying it
    # everywhere is the right split; the legged families are then a genuine held-out check.
    pool = {k: [] for k in ("ndl", "ndh", "obs")}
    mat = None
    wd = ws = None
    for nm in ["PointmazeMediumNavigate", "PointmazeLargeNavigate", "PointmazeGiantNavigate",
               "PointmazeTeleportNavigate"]:
        try:
            e = se.make(nm, batch=8, device="cuda")
            rows = agent_pixels(e, e.sample_states(96))
            sc = e._scene()
            mat = np.asarray(e._var(None, "agent.color"))
            wd, ws = np.asarray(sc.lights.diffuse), np.asarray(sc.lights.specular)
            r = dict(env=nm, n_px=int(len(rows["obs"])), fit=fit(rows, mat, wd, ws))
            for k in pool:
                pool[k].append(rows[k])
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {nm}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        res.append(r)
        log.write(json.dumps(r) + "\n")
        log.flush()
    if pool["obs"]:
        merged = {k: np.concatenate(v) for k, v in pool.items()}
        r = dict(env="POOLED", n_px=int(len(merged["obs"])), fit=fit(merged, mat, wd, ws))
        res.append(r)
        log.write(json.dumps(r) + "\n")
        log.flush()
    log.close()
    with open(os.path.join(OUT, "calibrate_agent.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
