"""Probe 3: WHICH geometry is drawn at the 179 pixels that move with the robot while every mover is
hidden? Segmentation answers it directly instead of by hypothesis.

Also dumps the frames as PNG so the region can be looked at rather than reasoned about.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_envs.render import mjwarp as MW
from probe1_model import free_cam_pose

RES = 480
OUT = "/mnt/vast/home/randall.balestriero/se_logs/kitchen"


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

    print("== lights ==")
    for i in range(m.nlight):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or f"light{i}"
        b = int(m.light_bodyid[i])
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"  {i} {nm!r:18s} type={int(m.light_type[i])} castshadow={int(m.light_castshadow[i])} "
              f"body={bn!r} weldid={int(m.body_weldid[b])}")

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vmovers = [g for g in vis if g not in static]

    env = gymnasium.make("FrankaKitchen-v1")
    u = env.unwrapped
    env.reset(seed=0)
    q0 = np.array(u.robot_env.data.qpos, np.float64)

    def shot(q, hide=None, seg=False):
        keep = np.array(m.geom_group, copy=True)
        if hide is not None:
            m.geom_group[hide] = 5
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        if seg:
            r.enable_segmentation_rendering()
        else:
            r.disable_segmentation_rendering()
        r.update_scene(d, camera=camid)
        img = r.render().copy()
        r.disable_segmentation_rendering()
        m.geom_group[:] = keep
        return img

    a = shot(q0, hide=vmovers)
    q1 = q0.copy(); q1[3] += 0.5
    b = shot(q1, hide=vmovers)
    dd = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    mask = dd > 0
    print(f"\n== the {int(mask.sum())} moving pixels, movers hidden, qpos[3]+0.5 ==")

    sg = shot(q0, hide=vmovers, seg=True)
    print(f"  segmentation array shape={sg.shape} dtype={sg.dtype}")
    ids = sg[..., 0][mask]
    types = sg[..., 1][mask]
    from collections import Counter
    print(f"  object ids at those pixels: {Counter(ids.tolist()).most_common(10)}")
    print(f"  object types at those pixels: {Counter(types.tolist()).most_common(10)}")
    for oid, n in Counter(ids.tolist()).most_common(10):
        if oid < 0:
            print(f"    id {oid}: {n} px  (nothing / skybox)")
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(oid))
        bd = int(m.geom_bodyid[int(oid)]) if int(oid) < m.ngeom else -1
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bd)
        print(f"    geom {oid} {nm!r} body={bn!r} weldid={int(m.body_weldid[bd])} "
              f"type={int(m.geom_type[int(oid)])}: {n} px")

    # does turning shadows off make the plate constant?
    print("\n== with shadows disabled on the plate ==")
    keep_shadow = int(m.vis.quality.shadowsize)
    m.vis.quality.shadowsize = 0
    a2 = shot(q0, hide=vmovers)
    b2 = shot(q1, hide=vmovers)
    dd2 = np.abs(a2.astype(np.int16) - b2.astype(np.int16)).max(-1)
    print(f"  max={int(dd2.max())} frac={float((dd2>0).mean()):.6f}")
    m.vis.quality.shadowsize = keep_shadow

    try:
        import imageio.v2 as iio
        iio.imwrite(f"{OUT}/plate_a.png", a)
        iio.imwrite(f"{OUT}/plate_b.png", b)
        vizmask = np.zeros((RES, RES, 3), np.uint8)
        vizmask[mask] = [255, 0, 0]
        iio.imwrite(f"{OUT}/plate_mask.png", np.maximum(a // 2, vizmask))
        iio.imwrite(f"{OUT}/full_a.png", shot(q0))
        print(f"  wrote pngs to {OUT}")
    except Exception as e:
        print(f"  png dump skipped: {e}")


if __name__ == "__main__":
    main()
