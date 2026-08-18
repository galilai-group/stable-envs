"""Probe 11: re-take the camera check in SEPARATE PROCESSES, and measure the frame share of each
mujoco_warp sub-limit.

probe1 compared our ``se:cam`` render against the official env's render inside ONE process and got
1.000000 exact. That number cannot be trusted as taken: ``gymnasium``'s ``OffScreenViewer.render``
never binds its GL context, so with two live viewers both renders can resolve against whichever EGL
context was created last -- and "both frames identical" is exactly what that failure looks like. Here
each render happens in its own process and the frames meet only as .npy files.

Also measures, because the two mujoco_warp sub-limits have different fixability:
  * share of the frame drawn by MOVING geoms whose UVs mujoco_warp cannot derive (partially
    recoverable by re-expressing as a mesh with cube texcoords);
  * share drawn by geoms with alpha < 1 (recoverable by nothing);
  * and how much of the frame is static and therefore bakeable exactly.
"""
from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_envs.render import mjwarp as MW

RES = 480
OUT = "/mnt/vast/home/randall.balestriero/se_logs/kitchen"


def qref():
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make("FrankaKitchen-v1")
    env.reset(seed=0)
    q = np.array(env.unwrapped.robot_env.data.qpos, np.float64)
    env.close()
    return q


def official():
    """ONE process, ONE viewer: the official env's own frame."""
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    q = qref()
    env = gymnasium.make("FrankaKitchen-v1", render_mode="rgb_array", width=RES, height=RES)
    u = env.unwrapped
    env.reset(seed=0)
    u.robot_env.data.qpos[:] = q
    u.robot_env.data.qvel[:] = 0
    mujoco.mj_forward(u.robot_env.model, u.robot_env.data)
    img = np.ascontiguousarray(u.render())
    c = u.robot_env.mujoco_renderer.viewer.cam
    np.save(f"{OUT}/cam_official.npy", img)
    print(f"official: unique colours={len(np.unique(img.reshape(-1,3), axis=0))} "
          f"min={img.min()} max={img.max()} mean={img.mean():.2f}")
    print(f"official cam: type={int(c.type)} fixed={int(c.fixedcamid)} dist={float(c.distance)} "
          f"az={float(c.azimuth)} el={float(c.elevation)} lookat={np.asarray(c.lookat).tolist()}")


def ours():
    """ONE process, ONE viewer: our recovered se:cam through a plain mujoco.Renderer."""
    from probe1_model import free_cam_pose
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    base = mujoco.MjModel.from_xml_path(xml)
    d0 = mujoco.MjData(base)
    mujoco.mj_forward(base, d0)
    pos, quat = free_cam_pose(base, d0, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"])
    spec = mujoco.MjSpec.from_file(xml)
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(base.vis.global_.fovy)
    m = spec.compile()
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    d = mujoco.MjData(m)
    d.qpos[:] = qref()
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    camid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")
    r = mujoco.Renderer(m, width=RES, height=RES)
    r.update_scene(d, camera=camid)
    img = r.render().copy()
    np.save(f"{OUT}/cam_ours.npy", img)
    print(f"ours: unique colours={len(np.unique(img.reshape(-1,3), axis=0))} "
          f"min={img.min()} max={img.max()} mean={img.mean():.2f}")


def shares():
    """Frame share of each sub-limit, from MuJoCo's own segmentation buffer (id, type)."""
    from probe1_model import free_cam_pose
    from probe5_contamination import pool
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    base = mujoco.MjModel.from_xml_path(xml)
    d0 = mujoco.MjData(base)
    mujoco.mj_forward(base, d0)
    pos, quat = free_cam_pose(base, d0, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"])
    spec = mujoco.MjSpec.from_file(xml)
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(base.vis.global_.fovy)
    m = spec.compile()
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    d = mujoco.MjData(m)
    camid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")
    r = mujoco.Renderer(m, width=RES, height=RES)

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vmovers = set(vis) - set(static)
    gaps = {g for g, _, _ in MW.texture_gaps(m)}
    alpha = MW.effective_alpha(m, "geom")
    trans = {g for g in vis if alpha[g] < 1.0}
    gap_movers = np.array(sorted(gaps & vmovers), np.int64)
    gap_static = np.array(sorted(gaps - vmovers), np.int64)
    mover_arr = np.array(sorted(vmovers), np.int64)
    static_vis = np.array(sorted(set(vis) & set(static)), np.int64)
    trans_arr = np.array(sorted(trans), np.int64)

    acc = {k: [] for k in ("static", "mover", "gap_mover", "gap_static", "alpha", "site", "sky")}
    for q in pool(12):
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=camid)
        sg = r.render().copy()
        r.disable_segmentation_rendering()
        oid, otype = sg[..., 0], sg[..., 1]     # (id, type) -- verified empirically in probe3
        isg = otype == int(mujoco.mjtObj.mjOBJ_GEOM)
        acc["static"].append(float((isg & np.isin(oid, static_vis)).mean()))
        acc["mover"].append(float((isg & np.isin(oid, mover_arr)).mean()))
        acc["gap_mover"].append(float((isg & np.isin(oid, gap_movers)).mean()))
        acc["gap_static"].append(float((isg & np.isin(oid, gap_static)).mean()))
        acc["alpha"].append(float((isg & np.isin(oid, trans_arr)).mean()))
        acc["site"].append(float((otype == int(mujoco.mjtObj.mjOBJ_SITE)).mean()))
        acc["sky"].append(float((oid < 0).mean()))
    print("\n== frame share, mean over 12 states the env reaches ==")
    for k, v in acc.items():
        print(f"  {k:11s} {np.mean(v):.4f}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    if a == "official":
        official()
    elif a == "ours":
        ours()
    elif a == "shares":
        shares()
    else:
        for step in ("official", "ours", "shares"):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), step],
                               capture_output=True, text=True, timeout=1800)
            print(p.stdout.rstrip() or p.stderr[-1500:])
        A = np.load(f"{OUT}/cam_official.npy")
        B = np.load(f"{OUT}/cam_ours.npy")
        dd = np.abs(A.astype(np.int16) - B.astype(np.int16)).max(-1)
        print(f"\n== se:cam vs official free camera, SEPARATE PROCESSES ==")
        print(f"  exact={float((dd==0).mean()):.6f} mean={float(dd.mean()):.4f} max={int(dd.max())}")
