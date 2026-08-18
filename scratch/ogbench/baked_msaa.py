"""The honest end-to-end number: baked static + analytic agent versus MuJoCo AT ITS DEFAULT SETTINGS.

``general_check.py`` scored against ``offsamples=0``. But OGBench renders with MuJoCo's default
``offsamples=4``, so every frame in every visual OGBench dataset is 4x multisampled, and MSAA coverage
is decided by GPU-specific sample positions. Two variants are scored here:

  ss=1   hard agent silhouette (a pixel is agent or it is not)
  ss=4   the agent's coverage estimated by 4x4 supersampling and alpha-blended over the baked plate

and, for reference, MuJoCo-vs-MuJoCo across a second render, so the MSAA-attributable disagreement can
be separated from a modelling error. Also re-hashes a fixed state sequence so the caller can compare
across nodes.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from general_check import make2, agent_geoms, cast_subset  # noqa: E402
from analytic_rgb import base_rgb, stats  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
# fitted on the pixels themselves (fit2.py): ambient / diffuse / specular / exponent
AMB, DIFF, SPEC, SHINE = 0.09804, 1.3907, 0.26, 60.0
L = np.array([0.0, 0.0, 1.0])


def agent_layer(m, d, cam, img, ag, ss):
    """(coverage, colour) of the agent at ss-times supersampling, box-averaged back to img."""
    R = img * ss
    gid, t, o, dr = cast_subset(m, d, cam, R, R, ag)
    vis = t < INF / 2
    col = np.zeros((R, R, 3))
    if vis.any():
        p = o + t[..., None] * dr
        n = normals(m, d, np.where(vis, gid, -1), p)
        V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
        hv = L + V
        hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)
        S = AMB + DIFF * np.clip(n @ L, 0, None) + SPEC * np.clip((n * hv).sum(-1), 0, None) ** SHINE
        base = np.zeros((R, R, 3))
        for g in np.unique(gid[vis]):
            base[vis & (gid == g)] = base_rgb(m, int(g))
        col = np.clip(base * S[..., None], 0, 1) * 255.0
    cov = vis.astype(np.float64)
    if ss > 1:
        col = (col * cov[..., None]).reshape(img, ss, img, ss, 3).sum((1, 3))
        cov = cov.reshape(img, ss, img, ss).sum((1, 3))
        col = np.where(cov[..., None] > 0, col / np.maximum(cov[..., None], 1e-9), 0.0)
        cov = cov / (ss * ss)
    return cov, col, t


def run(loco, size, img=224, nstates=12):
    import mujoco
    import imageio.v2 as imageio
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    m.vis.quality.offsamples = 4                       # MuJoCo/OGBench default
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    ag, names = agent_geoms(m)
    static = [g for g in range(m.ngeom) if g not in ag]

    keep = m.geom_rgba[ag].copy()
    m.geom_rgba[ag, 3] = 0.0
    r2 = mujoco.Renderer(m, width=img, height=img)
    r2.update_scene(d, camera=u.custom_camera)
    plate = r2.render().astype(np.float64)
    m.geom_rgba[ag] = keep
    _, bg_t, _, _ = cast_subset(m, d, cam, img, img, static)

    acc = {"ss1": [], "ss4": [], "mujoco_self": []}
    for s in range(nstates):
        for _ in range(11):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        ref = r.render().copy()
        r.update_scene(d, camera=u.custom_camera)
        ref_b = r.render().copy()
        acc["mujoco_self"].append(stats(ref, ref_b))
        for ss, key in ((1, "ss1"), (4, "ss4")):
            cov, col, t = agent_layer(m, d, cam, img, ag, ss)
            occl = np.repeat(np.repeat(bg_t, 1, 0), 1, 1)
            cov = np.where(t.reshape(img, ss, img, ss).min((1, 3)) < occl if ss > 1
                           else t < occl, cov, 0.0)
            out = plate * (1 - cov[..., None]) + col * cov[..., None]
            mine = np.clip(np.round(out), 0, 255).astype(np.uint8)
            acc[key].append(stats(mine, ref))
            if s == 0 and ss == 4:
                imageio.imwrite(os.path.join(OUT, f"msaa_{loco}_{size}.png"),
                                np.concatenate([ref, mine,
                                                np.clip(np.abs(mine.astype(np.int16) -
                                                               ref.astype(np.int16)) * 8, 0, 255)
                                                .astype(np.uint8)], 1))
    env.close()
    return dict(env=f"{loco}maze-{size}", offsamples=4, nstates=nstates,
                **{k: {kk: round(float(np.mean([x[kk] for x in v])), 6) for kk in v[0]}
                   for k, v in acc.items()})


def hashes(loco="point", size="large", img=224, n=12):
    env, u = make2(loco, size, img)
    m = u.model
    rng = np.random.default_rng(7)
    hs = []
    for _ in range(n):
        qpos = np.asarray(u.init_qpos, np.float64).copy()
        qpos[:2] = [rng.uniform(0, 36), rng.uniform(0, 24)]
        if m.nq > 2:
            qpos[2:] += rng.normal(0, 0.02, m.nq - 2)
        u.set_state(qpos, np.zeros(m.nv))
        hs.append(hashlib.sha256(np.ascontiguousarray(u.render())).hexdigest()[:16])
    env.close()
    return hs


def main():
    os.makedirs(OUT, exist_ok=True)
    host = socket.gethostname()
    out = {"host": host, "hashes_pointmaze_large": hashes()}
    print(json.dumps({"host": host, "hashes": out["hashes_pointmaze_large"]}), flush=True)
    out["msaa"] = []
    for loco, size in [("point", "medium"), ("point", "large"), ("ant", "large"),
                       ("humanoid", "large")]:
        try:
            r = run(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        out["msaa"].append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, f"msaa_{host}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
