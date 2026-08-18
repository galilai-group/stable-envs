"""Model census for every HumanoidBench scene: the four numbers that decide the port.

nmesh / ntex (by type) / casting lights / textured-primitive share, plus the camera mode and the
group-0-2 geom breakdown. Pure MuJoCo, no rendering, no GPU: this is the cheap gate.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ASSETS = "/mnt/vast/home/randall.balestriero/src/humanoid-bench/humanoid_bench/assets"
OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
GROUPS = (0, 1, 2)
TEXTURED_OK = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_MESH))


def census(path: str) -> dict:
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in GROUPS]
    types = Counter(int(m.geom_type[g]) for g in vis)

    # textured geoms MJWarp cannot UV: not PLANE and not MESH, wearing a material with a texture
    bad = []
    for g in vis:
        mid = int(m.geom_matid[g])
        if mid < 0 or int(m.geom_type[g]) in TEXTURED_OK:
            continue
        t = m.mat_texid[mid]
        t = int(np.max(t)) if np.ndim(t) else int(t)
        if t >= 0:
            bad.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")

    tex_types = Counter(int(m.tex_type[t]) for t in range(m.ntex))
    cast = int(np.sum(np.asarray(m.light_castshadow, np.int64) != 0)) if m.nlight else 0

    # alpha < 1 in the visible groups (MJWarp has no alpha path)
    rgba = np.asarray(m.geom_rgba, np.float64)
    matid = np.asarray(m.geom_matid, np.int64)
    alpha = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    take = default & (matid >= 0)
    if take.any():
        alpha[take] = np.asarray(m.mat_rgba, np.float64)[matid[take], 3]
    translucent = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                   for g in vis if 0.0 < alpha[g] < 1.0]
    invisible = [g for g in vis if alpha[g] <= 0.0]

    cams = {}
    for c in range(m.ncam):
        cams[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or f"cam{c}"] = int(m.cam_mode[c])

    sites_vis = [s for s in range(m.nsite) if int(m.site_group[s]) in GROUPS]

    return dict(
        nq=int(m.nq), nv=int(m.nv), ngeom=int(m.ngeom), nvisible=len(vis),
        nmesh=int(m.nmesh), ntex=int(m.ntex), tex_types={str(k): v for k, v in tex_types.items()},
        nlight=int(m.nlight), casting_lights=cast,
        geom_types={str(k): v for k, v in types.items()},
        n_mesh_geoms=types.get(int(mujoco.mjtGeom.mjGEOM_MESH), 0),
        textured_primitives=bad, n_textured_primitives=len(bad),
        translucent=translucent, n_translucent=len(translucent), n_zero_alpha=len(invisible),
        cameras=cams, nsite_visible=len(sites_vis),
        shadowsize=int(m.vis.quality.shadowsize), offsamples=int(m.vis.quality.offsamples),
        mat_reflectance_max=float(np.max(m.mat_reflectance)) if m.nmat else 0.0,
        geom_margin_nonzero=int(np.sum(np.asarray(m.geom_margin) > 0)),
        nmat=int(m.nmat),
    )


def main():
    files = sorted(f for f in os.listdir(os.path.join(ASSETS, "envs")) if f.endswith(".xml"))
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    out = {}
    for f in files:
        if only and not f.startswith(only):
            continue
        try:
            out[f[:-4]] = census(os.path.join(ASSETS, "envs", f))
        except Exception as e:                                        # pragma: no cover
            out[f[:-4]] = {"error": f"{type(e).__name__}: {e}"}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe1_facts.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    hdr = f"{'scene':34s} {'nq':>4s} {'vis':>4s} {'mesh':>5s} {'ntex':>5s} {'cast':>4s} {'texprim':>7s} {'alpha<1':>7s} {'sites':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for k, v in out.items():
        if "error" in v:
            print(f"{k:34s} {v['error']}")
            continue
        print(f"{k:34s} {v['nq']:4d} {v['nvisible']:4d} {v['nmesh']:5d} {v['ntex']:5d} "
              f"{v['casting_lights']:4d} {v['n_textured_primitives']:7d} {v['n_translucent']:7d} "
              f"{v['nsite_visible']:5d}")


if __name__ == "__main__":
    main()
