"""The free control, finished: what does a camera SEE, and is a triangle rasterizer the right shape?

Two questions, one geometry pass, no GPU renderer involved.

**1. Why is the wrist camera 2.3x faster than agentview on the same kitchen?**
``probe_decompose.py`` proved the scene half: the two cells compile the SAME model -- identical geom,
mesh, triangle and texture counts -- and differ only in ``cam_id``, so nothing about the scene can
explain it and the whole delta is inside the ``render`` megakernel. This measures the camera half.
A ray tracer's per-ray cost is the number of bounding volumes it has to open before it can name the
nearest hit, so the quantity to count is, for a ray through each pixel: how many of the scene's geom
AABBs does it pierce, and how many of those lie in FRONT of the surface it eventually lands on. The
second number is the one a BVH actually pays, because a traversal that has already found a hit at
distance ``t`` culls every box beyond it.

**2. Is a triangle rasterizer even the right algorithm here?**
A rasterizer's cost is Theta(triangles submitted) + Theta(fragments); a ray tracer's is
Theta(pixels x traversal depth). Which wins is decided by the ratio of triangles to pixels, and these
scenes are 256x256 = 65,536 pixels against 295k-668k triangles. So this also reports, per camera:
triangles inside the frustum, their projected area in pixels, and what fraction of them are smaller
than one pixel -- geometry that a rasterizer must transform, bin and clip in full and that a ray
tracer never opens.

Pure NumPy on the CPU, on the same compiled model the GPU path renders, so it can be checked by hand.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np


def camera_rays(m, d, cam, res, rng, n_rays):
    """``(origin (3,), dirs (n,3))`` for ``n_rays`` random pixels of a ``res x res`` frame."""
    pos = np.asarray(d.cam_xpos[cam], np.float64)
    R = np.asarray(d.cam_xmat[cam], np.float64).reshape(3, 3)
    fovy = float(m.cam_fovy[cam])
    half = np.tan(np.deg2rad(fovy) / 2.0)
    if n_rays >= res * res:
        yy, xx = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
        px, py = xx.ravel(), yy.ravel()
    else:
        px = rng.integers(0, res, n_rays)
        py = rng.integers(0, res, n_rays)
    # MuJoCo's camera looks down local -z, +y up, +x right; pixel (0,0) is the top-left.
    sx = (2.0 * (px + 0.5) / res - 1.0) * half
    sy = (1.0 - 2.0 * (py + 0.5) / res) * half
    local = np.stack([sx, sy, -np.ones_like(sx)], 1)
    dirs = local @ R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return pos, dirs, px, py


def geom_aabbs(m, d, ids):
    """World AABBs of ``ids``, from ``geom_rbound`` about ``geom_xpos`` (what a top-level BVH holds)."""
    c = np.asarray(d.geom_xpos, np.float64)[ids]
    r = np.asarray(m.geom_rbound, np.float64)[ids]
    # A plane has rbound 0 and is unbounded; give it the model's own extent so it is not silently
    # excluded from the count -- the floor is the one surface every locomotion ray hits.
    ext = float(np.abs(np.asarray(d.geom_xpos, np.float64)).max() + 50.0)
    r = np.where(r <= 0, ext, r)
    return c - r[:, None], c + r[:, None]


def slab_hits(origin, dirs, lo, hi):
    """``(n_rays, n_boxes)`` entry distance, or +inf where the ray misses. Standard slab test."""
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t0 = (lo[None] - origin[None, None]) * inv[:, None, :]
    t1 = (hi[None] - origin[None, None]) * inv[:, None, :]
    tmin = np.minimum(t0, t1).max(-1)
    tmax = np.maximum(t0, t1).min(-1)
    ok = (tmax >= np.maximum(tmin, 0.0))
    return np.where(ok, np.maximum(tmin, 0.0), np.inf)


def project_triangles(m, d, ids, cam, res):
    """Per-frustum triangle census for the mesh geoms in ``ids``."""
    import mujoco
    pos = np.asarray(d.cam_xpos[cam], np.float64)
    R = np.asarray(d.cam_xmat[cam], np.float64).reshape(3, 3)
    half = np.tan(np.deg2rad(float(m.cam_fovy[cam])) / 2.0)
    f = res / (2.0 * half)
    n_in, n_tot, areas = 0, 0, []
    for g in ids:
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        did = int(m.geom_dataid[g])
        if did < 0:
            continue
        va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fa, nf = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
        v = np.asarray(m.mesh_vert[va:va + nv], np.float64).reshape(nv, 3)
        fc = np.asarray(m.mesh_face[fa:fa + nf], np.int64).reshape(nf, 3)
        gp = np.asarray(d.geom_xpos[g], np.float64)
        gm = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
        w = v @ gm.T + gp
        c = (w - pos) @ R                                   # into camera frame
        z = -c[:, 2]
        u = f * c[:, 0] / np.where(z > 1e-6, z, 1e-6) + res / 2.0
        vv = -f * c[:, 1] / np.where(z > 1e-6, z, 1e-6) + res / 2.0
        tu, tv, tz = u[fc], vv[fc], z[fc]
        infront = (tz > 1e-6).all(1)
        inside = ((tu >= 0) & (tu < res) & (tv >= 0) & (tv < res)).any(1)
        keep = infront & inside
        ar = 0.5 * np.abs((tu[:, 1] - tu[:, 0]) * (tv[:, 2] - tv[:, 0])
                          - (tu[:, 2] - tu[:, 0]) * (tv[:, 1] - tv[:, 0]))
        n_tot += nf
        n_in += int(keep.sum())
        areas.append(ar[keep])
    ar = np.concatenate(areas) if areas else np.zeros(0)
    return {"tri_total": int(n_tot), "tri_in_frustum": int(n_in),
            "tri_subpixel_frac": float((ar < 1.0).mean()) if ar.size else None,
            "tri_area_mean_px": float(ar.mean()) if ar.size else None,
            "tri_area_median_px": float(np.median(ar)) if ar.size else None,
            "tri_area_sum_px": float(ar.sum()) if ar.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--rays", type=int, default=8192)
    ap.add_argument("--states", type=int, default=4)
    a = ap.parse_args()

    import mujoco
    import stable_envs as se
    cls = se.REGISTRY[a.env]
    if a.env.startswith("RoboCasa"):
        from stable_envs.envs import robocasa as M
        s = M._scene(cls.KITCHEN)
        m, groups = s.model, M.VISIBLE_GROUPS
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        import torch
        st = env.sample_states(a.states)
        qpos = env._full_qpos(torch.as_tensor(st, device="cuda")).cpu().numpy().astype(np.float64)
    else:
        from stable_envs.envs import humanoidbench as M
        b = M._build_bundle(cls.ROBOT, cls.TASK)
        m, groups, cam = b.model, M.VISIBLE_GROUPS, int(b.info["cam"])
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        qpos = np.asarray(env.sample_states(a.states), np.float64)
    del env

    ids = np.where(np.isin(np.asarray(m.geom_group, np.int64), list(groups)))[0]
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    out = {"env": a.env, "camera": cls.CAMERA if hasattr(cls, "CAMERA") else None,
           "res": a.res, "n_visible_geoms": int(len(ids)), "rows": []}

    for si, row in enumerate(qpos):
        d.qpos[:] = row
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        origin, dirs, px, py = camera_rays(m, d, cam, a.res, rng, a.rays)
        lo, hi = geom_aabbs(m, d, ids)
        t = slab_hits(origin, dirs, lo, hi)                 # (rays, boxes)
        pierced = np.isfinite(t).sum(1)

        # The surface each ray actually lands on, from MuJoCo's own ray caster, so "in front of the
        # hit" is the renderer's own answer and not a re-derivation of it.
        gid = np.zeros(1, np.int32)
        hitdist = np.empty(len(dirs))
        geomgroup = np.zeros(6, np.uint8)
        for gg in groups:
            geomgroup[gg] = 1
        for i in range(len(dirs)):
            hitdist[i] = mujoco.mj_ray(m, d, origin, dirs[i], geomgroup, 1, -1, gid)
        miss = hitdist < 0
        far = np.where(miss, np.inf, hitdist)
        before = (t < far[:, None]).sum(1)

        r = {"state": si, "miss_frac": float(miss.mean()),
             "hit_dist_mean": float(hitdist[~miss].mean()) if (~miss).any() else None,
             "hit_dist_median": float(np.median(hitdist[~miss])) if (~miss).any() else None,
             "aabb_pierced_mean": float(pierced.mean()),
             "aabb_pierced_p90": float(np.percentile(pierced, 90)),
             "aabb_before_hit_mean": float(before.mean()),
             "aabb_before_hit_p90": float(np.percentile(before, 90)),
             "n_rays": int(len(dirs))}
        r.update(project_triangles(m, d, ids, cam, a.res))
        out["rows"].append(r)
        print(json.dumps(r), flush=True)

    keys = [k for k in out["rows"][0] if isinstance(out["rows"][0][k], (int, float))]
    out["mean"] = {k: float(np.mean([r[k] for r in out["rows"] if r[k] is not None]))
                   for k in keys}
    print(json.dumps({"env": a.env, "mean": out["mean"]}), flush=True)
    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/rays_{a.env}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
