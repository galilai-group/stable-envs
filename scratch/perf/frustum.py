"""A kitchen's ray-cast pays for geometry no ray can reach. How much, and is dropping it exact?

RoboCasa's single light has ``castshadow = False`` (measured: ``use_shadows=False`` changes the frame
by 0/255 and the time by 0.0%), and ``mujoco_warp``'s shader has no reflection, no refraction and no
secondary bounce. So the ONLY way a geom reaches a pixel is by being hit by a primary ray, and every
primary ray lies inside the camera frustum: a geom whose bounding sphere is entirely outside that
frustum cannot change any pixel, and removing it from the BVH is exact rather than approximate.

Two conditions are needed for it to STAY exact and both are checked here:

* the geom must be WELDED TO THE WORLD, so no state can move it back into view. A free-joint object on
  a counter is culled by its build-time pose and would vanish the moment a state slid it into frame.
* the frustum must be the union over every camera the variation space can produce, not the default
  one. ``camera.offset``, ``camera.pan`` and ``camera.tilt`` are Boxes, so the bound is taken over
  their corners.

Reports the cull fraction, the speed, and the max abs difference against the uncalled render.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se


def frustum_planes(cam_pos, cam_mat, fovy_deg, aspect):
    """Inward-pointing ``(5, 4)`` planes (nx, ny, nz, d) with ``n . x + d >= 0`` inside the frustum.

    MuJoCo's camera looks down its local -z with +y up, so the four side planes are the local axes
    tilted toward -z by the half angles. The fifth is the camera plane itself rather than the near
    plane: keeping everything in front of the camera is the conservative choice and costs a handful of
    geoms that a znear offset would have dropped.
    """
    t = np.tan(np.deg2rad(fovy_deg) / 2.0)
    fwd, up, right = -cam_mat[:, 2], cam_mat[:, 1], cam_mat[:, 0]
    out = []
    for n_local in (right + t * aspect * fwd, -right + t * aspect * fwd,
                    up + t * fwd, -up + t * fwd, fwd):
        n = n_local / np.linalg.norm(n_local)
        out.append(np.concatenate([n, [-np.dot(n, cam_pos)]]))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.05, help="metres of slack on every plane")
    a = ap.parse_args()

    import mujoco

    from stable_envs.envs import robocasa as RC

    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    s = RC._scene(cls.KITCHEN)
    m = s.model
    d = mujoco.MjData(m)
    d.qpos[:] = s.pinned
    mujoco.mj_forward(m, d)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)

    space = env.make_variation_space()
    off_lo = np.asarray(space["camera"]["offset"].low, np.float64)
    off_hi = np.asarray(space["camera"]["offset"].high, np.float64)
    pan_lo, pan_hi = float(space["camera"]["pan"].low[0]), float(space["camera"]["pan"].high[0])
    tilt_lo, tilt_hi = float(space["camera"]["tilt"].low[0]), float(space["camera"]["tilt"].high[0])

    # Every corner of the knob box, so the kept set is a superset of what any style can see.
    keep = np.zeros(m.ngeom, bool)
    znear = m.vis.map.znear * m.stat.extent
    corners = list(itertools.product(*zip(off_lo, off_hi))), (pan_lo, pan_hi), (tilt_lo, tilt_hi)
    ncam = 0
    for o in corners[0]:
        for p in corners[1]:
            for t in corners[2]:
                pos, quat = env._camera_pose(np.asarray([o], np.float64),
                                             np.asarray([[p]], np.float64),
                                             np.asarray([[t]], np.float64))
                R = np.empty(9)
                mujoco.mju_quat2Mat(R, quat[0])
                pl = frustum_planes(pos[0], R.reshape(3, 3), float(m.cam_fovy[cam]), 1.0)
                for g in range(m.ngeom):
                    if keep[g]:
                        continue
                    c = np.asarray(d.geom_xpos[g], np.float64)
                    r = float(m.geom_rbound[g]) + a.margin
                    if all(np.dot(q[:3], c) + q[3] >= -r for q in pl):
                        keep[g] = True
                ncam += 1

    movable = np.asarray([int(m.body_weldid[m.geom_bodyid[g]]) != 0 for g in range(m.ngeom)])
    visible = np.isin(m.geom_group, list(RC.VISIBLE_GROUPS))
    cull = visible & ~keep & ~movable
    print(f"# {a.env}: {ncam} frusta, {int(visible.sum())} visible geoms, "
          f"{int((visible & movable).sum())} of them movable, "
          f"CULLABLE {int(cull.sum())} ({100*cull.sum()/max(1,visible.sum()):.1f}%)", flush=True)

    def build_and_time(cull_mask):
        RC._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        keep_groups = np.array(m.geom_group, copy=True)
        if cull_mask is not None and cull_mask.any():
            m.geom_group[cull_mask] = RC.MW.HIDDEN_GROUP
        try:
            c = RC._context(cls.KITCHEN, cls.CAMERA, a.size, a.nworld)
            for _ in range(3):
                c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            ts = []
            for _ in range(15):
                t0 = time.perf_counter()
                c.wp.capture_launch(c.graph)
                c.wp.synchronize()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts)), c.bgra[:a.nworld][..., [2, 1, 0]].clone()
        finally:
            m.geom_group[:] = keep_groups
            RC._CTX_CACHE.clear()

    t0, f0 = build_and_time(None)
    t1, f1 = build_and_time(cull)
    diff = int((f0.to(torch.int16) - f1.to(torch.int16)).abs().max())
    npx = int(((f0.to(torch.int16) - f1.to(torch.int16)).abs().max(-1).values > 0).sum())
    print(f"baseline {t0*1e3:7.3f} ms  {a.nworld/t0:9,.0f} f/s", flush=True)
    print(f"culled   {t1*1e3:7.3f} ms  {a.nworld/t1:9,.0f} f/s  x{t0/t1:.2f}   "
          f"maxdiff {diff}/255 on {npx} px", flush=True)
    print("JSON " + json.dumps({"env": a.env, "cullable": int(cull.sum()),
                                "visible": int(visible.sum()), "base_ms": t0 * 1e3,
                                "culled_ms": t1 * 1e3, "speedup": t0 / t1, "maxdiff": diff,
                                "npix": npx}), flush=True)


if __name__ == "__main__":
    main()
