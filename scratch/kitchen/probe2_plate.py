"""Probe 2: why is the movers-hidden plate not constant, and which texture gaps are MOVERS?

Three questions:
  1. Is the plate render deterministic at all on this node? (same state, twice)
  2. Which qpos dimension moves the movers-hidden plate? (one dim at a time)
  3. Of the 48 textured non-mesh geoms MJWarp cannot UV, how many are MOVERS? Static ones are baked
     through MuJoCo and never reach MJWarp, so only the movers are a fidelity problem.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render import mjwarp as MW

RES = 480
from probe1_model import free_cam_pose


def main():
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    gymnasium.register_envs(gymnasium_robotics)

    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    m0 = mujoco.MjModel.from_xml_path(xml)
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    pos, quat = free_cam_pose(m0, d0, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"])

    spec = mujoco.MjSpec.from_file(xml)
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(m0.vis.global_.fovy)
    m = spec.compile()
    m.vis.quality.offsamples = int(m0.vis.quality.offsamples)
    d = mujoco.MjData(m)
    camid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")
    r = mujoco.Renderer(m, width=RES, height=RES)

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vmovers = [g for g in vis if g not in static]

    # --- 3. which texture gaps are movers?
    gaps = MW.texture_gaps(m)
    gap_ids = {g for g, _, _ in gaps}
    gap_movers = sorted(gap_ids & set(vmovers))
    gap_static = sorted(gap_ids - set(vmovers))
    print(f"== texture gaps ==")
    print(f"  total {len(gap_ids)}: {len(gap_static)} STATIC (baked, harmless), "
          f"{len(gap_movers)} MOVERS (MJWarp paints them one texel)")
    for g in gap_movers:
        b = int(m.geom_bodyid[g])
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        mid = int(m.geom_matid[g])
        mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) if mid >= 0 else None
        print(f"    geom {g} type={int(m.geom_type[g])} body={bn!r} material={mn!r}")

    def shot(q, hide=None):
        keep = np.array(m.geom_group, copy=True)
        if hide is not None:
            m.geom_group[hide] = 5
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camid)
        img = r.render().copy()
        m.geom_group[:] = keep
        return img

    env = gymnasium.make("FrankaKitchen-v1")
    u = env.unwrapped
    env.reset(seed=0)
    q0 = np.array(u.robot_env.data.qpos, np.float64)

    # --- 1. determinism
    a1 = shot(q0, hide=vmovers)
    a2 = shot(q0, hide=vmovers)
    print(f"\n== determinism (same state, movers hidden, twice) ==")
    print(f"  max={int(np.abs(a1.astype(np.int16)-a2.astype(np.int16)).max())}")

    # --- 2. per-dim plate sensitivity
    print(f"\n== plate sensitivity, one qpos dim at a time (movers hidden) ==")
    alt = {19: 0.37, 20: 0.0, 21: 1.45, 22: -0.75,
           17: -0.69, 18: -0.05, 9: -0.88, 10: -0.01, 11: -0.88, 12: -0.01,
           13: -0.92, 14: -0.01, 15: -0.92, 16: -0.01,
           23: -0.23, 24: 0.75, 25: 1.62, 26: 0.99, 27: 0.0, 28: 0.0, 29: -0.06}
    bad = []
    for k in range(m.nq):
        q = q0.copy()
        q[k] = alt.get(k, q0[k] + (0.5 if k < 9 else 0.3))
        b = shot(q, hide=vmovers)
        dd = np.abs(a1.astype(np.int16) - b.astype(np.int16)).max(-1)
        if dd.max() > 0:
            bad.append((k, int(dd.max()), float((dd > 0).mean())))
            print(f"  qpos[{k:2d}] -> max={int(dd.max()):4d} frac={float((dd>0).mean()):.6f}")
    if not bad:
        print("  none: the plate is a function of the appearance alone")

    # --- where are the offending pixels?
    if bad:
        k = bad[0][0]
        q = q0.copy()
        q[k] = alt.get(k, q0[k] + (0.5 if k < 9 else 0.3))
        b = shot(q, hide=vmovers)
        dd = np.abs(a1.astype(np.int16) - b.astype(np.int16)).max(-1)
        ys, xs = np.nonzero(dd)
        print(f"\n  qpos[{k}] diff bbox rows {ys.min()}..{ys.max()} cols {xs.min()}..{xs.max()} "
              f"({len(ys)} px)")
        np.save("/mnt/vast/home/randall.balestriero/se_logs/kitchen/plate_diff.npy", dd)

    # --- true mover silhouette vs 'anything changed'
    full = shot(q0)
    plate = a1
    changed = np.abs(full.astype(np.int16) - plate.astype(np.int16)).max(-1) > 0
    print(f"\n== movers-vs-plate at the reset state ==")
    print(f"  pixels the movers change: {float(changed.mean()):.4f}")


if __name__ == "__main__":
    main()
