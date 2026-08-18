"""Probe 1: what IS the Franka Kitchen scene, and can a plate bake possibly work here?

Answers, in order:
  * model census: ngeom / nmesh / nq / lights (and their castshadow) / materials (and reflectance)
  * texture_gaps: textured geoms whose UVs mujoco_warp cannot derive
  * static vs mover geoms, by body_weldid
  * THE decisive number: with the movers hidden, is the rest of the frame state-independent, and
    how many NON-MOVER pixels does the official render change by when a mover moves? That second
    quantity is exactly the shadow + reflection contamination a plate cannot carry.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

RES = 480


def free_cam_pose(m, d, lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, np.float64)
    cam.distance, cam.azimuth, cam.elevation = float(distance), float(azimuth), float(elevation)
    scn = mujoco.MjvScene(m, maxgeom=int(m.ngeom) + int(m.nsite) + 8)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), cam,
                           mujoco.mjtCatBit.mjCAT_STATIC, scn)
    e0, e1 = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.array(e0.pos, np.float64) + np.array(e1.pos, np.float64))
    fwd = np.array(e0.forward, np.float64); fwd /= np.linalg.norm(fwd)
    up = np.array(e0.up, np.float64)
    z = -fwd
    x = np.cross(fwd, up / np.linalg.norm(up)); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(-1))
    return pos, q


def main():
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    gymnasium.register_envs(gymnasium_robotics)

    from gymnasium_robotics.envs.franka_kitchen import franka_env
    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"== census ==")
    print(f"nq={m.nq} nv={m.nv} ngeom={m.ngeom} nmesh={m.nmesh} nbody={m.nbody} "
          f"nsite={m.nsite} nlight={m.nlight} ntex={m.ntex} nmat={m.nmat} ncam={m.ncam}")
    print(f"offsamples={m.vis.quality.offsamples} shadowsize={m.vis.quality.shadowsize} "
          f"fovy={m.vis.global_.fovy} offw={m.vis.global_.offwidth} offh={m.vis.global_.offheight}")
    print(f"camera cfg: {CC}")

    print("\n== lights ==")
    for i in range(m.nlight):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or f"light{i}"
        print(f"  {i} {nm:16s} castshadow={int(m.light_castshadow[i])} "
              f"type={int(m.light_type[i])} bodyid={int(m.light_bodyid[i])} "
              f"diffuse={np.round(m.light_diffuse[i],3).tolist()} "
              f"cutoff={float(m.light_cutoff[i]):.1f}")

    print("\n== materials with reflectance>0 ==")
    for i in range(m.nmat):
        if float(m.mat_reflectance[i]) > 0:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}"
            tex = m.mat_texid[i]
            tex = np.asarray(tex).reshape(-1)
            print(f"  {nm:24s} refl={float(m.mat_reflectance[i]):.2f} "
                  f"shin={float(m.mat_shininess[i]):.2f} texid={tex[tex>=0].tolist()}")

    print("\n== geom census (visible groups 0-2) ==")
    import sys
    sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
    from stable_envs.render import mjwarp as MW
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    print(f"  visible geoms: {len(vis)} / {m.ngeom}")
    gaps = MW.texture_gaps(m)
    print(f"  texture_gaps: {len(gaps)}")
    for g, nm, t in gaps:
        print(f"    geom {g} {nm!r} type={t}")
    alpha = MW.effective_alpha(m, "geom")
    trans = [g for g in vis if alpha[g] < 1.0]
    print(f"  translucent visible geoms: {len(trans)} -> "
          f"{[(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in trans]}")
    sa = MW.effective_alpha(m, "site")
    vsites = [s for s in range(m.nsite) if int(m.site_group[s]) in MW.DEFAULT_GROUPS]
    print(f"  visible sites: {len(vsites)} (alphas {np.round(sa[vsites],2).tolist() if vsites else []})")

    # geom types among visible
    from collections import Counter
    print(f"  visible geom types: {Counter(int(m.geom_type[g]) for g in vis)}")
    # textured visible geoms and their type
    ntex_geoms = 0
    for g in vis:
        mid = int(m.geom_matid[g])
        if mid >= 0:
            t = np.asarray(m.mat_texid[mid]).reshape(-1)
            if (t >= 0).any():
                ntex_geoms += 1
    print(f"  textured visible geoms: {ntex_geoms}")

    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    movers = [g for g in range(m.ngeom) if g not in static]
    vstatic = [g for g in static if g in vis]
    vmovers = [g for g in movers if g in vis]
    print(f"\n  STATIC geoms: {len(static)} ({len(vstatic)} visible)")
    print(f"  MOVER  geoms: {len(movers)} ({len(vmovers)} visible)")
    print(f"  mover names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in vmovers][:40]}")

    # ---------------- render experiments ----------------
    pos, quat = free_cam_pose(m, d, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"])
    print(f"\n== free camera pose ==\n  pos={np.round(pos,6).tolist()} quat={np.round(quat,6).tolist()}")

    # states: reset state, and a few states with the movers moved
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv
    env = gymnasium.make("FrankaKitchen-v1")
    u = env.unwrapped
    env.reset(seed=0)
    q0 = np.array(u.robot_env.data.qpos, np.float64)
    print(f"\n  reset qpos = {np.round(q0, 4).tolist()}")

    # add a camera so we render exactly what the official free camera sees
    spec = mujoco.MjSpec.from_file(xml)
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(m.vis.global_.fovy)
    m2 = spec.compile()
    m2.vis.quality.offsamples = int(m.vis.quality.offsamples)
    d2 = mujoco.MjData(m2)
    camid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")
    r = mujoco.Renderer(m2, width=RES, height=RES)

    def shot(q, hide=None):
        keep = np.array(m2.geom_group, copy=True)
        if hide is not None:
            m2.geom_group[hide] = 5
        d2.qpos[:] = q
        d2.qvel[:] = 0
        mujoco.mj_forward(m2, d2)
        r.update_scene(d2, camera=camid)
        img = r.render().copy()
        m2.geom_group[:] = keep
        return img

    # our own free-camera check: se:cam vs the official env's render
    official = None
    envr = gymnasium.make("FrankaKitchen-v1", render_mode="rgb_array", width=RES, height=RES)
    ur = envr.unwrapped
    envr.reset(seed=0)
    ur.robot_env.data.qpos[:] = q0
    ur.robot_env.data.qvel[:] = 0
    mujoco.mj_forward(ur.robot_env.model, ur.robot_env.data)
    official = np.ascontiguousarray(ur.render())
    c = ur.robot_env.mujoco_renderer.viewer.cam
    print(f"  official viewer cam: type={int(c.type)} fixed={int(c.fixedcamid)} "
          f"dist={float(c.distance)} az={float(c.azimuth)} el={float(c.elevation)} "
          f"lookat={np.asarray(c.lookat).tolist()}")

    ours = shot(q0)
    dd = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    print(f"  se:cam vs official free camera: exact={float((dd==0).mean()):.6f} "
          f"mean={float(dd.mean()):.4f} max={int(dd.max())}")

    # plate viability: hide movers, two very different states
    rng = np.random.default_rng(0)
    qa = q0.copy()
    qb = q0.copy()
    qb[:9] = q0[:9] + rng.uniform(-0.5, 0.5, 9)
    qb[19] = 0.37          # slide cabinet open
    qb[21] = 1.45          # right hinge cabinet open
    qb[22] = -0.75         # microwave open
    qb[23:26] = [-0.23, 0.75, 1.62]  # kettle moved

    pa = shot(qa, hide=vmovers)
    pb = shot(qb, hide=vmovers)
    dp = np.abs(pa.astype(np.int16) - pb.astype(np.int16)).max(-1)
    print(f"\n== plate constancy (movers hidden, two states) ==")
    print(f"  max={int(dp.max())} nonzero_frac={float((dp>0).mean()):.6f}")

    fa = shot(qa)
    fb = shot(qb)
    # mover silhouette per state, from MuJoCo itself: pixels that change when movers are hidden
    sa_mask = np.abs(fa.astype(np.int16) - pa.astype(np.int16)).max(-1) > 0
    sb_mask = np.abs(fb.astype(np.int16) - pb.astype(np.int16)).max(-1) > 0
    print(f"\n== 'movers changed something' fraction ==")
    print(f"  state A: {float(sa_mask.mean()):.4f}   state B: {float(sb_mask.mean()):.4f}")

    np.save("/mnt/vast/home/randall.balestriero/se_logs/kitchen/fa.npy", fa)
    np.save("/mnt/vast/home/randall.balestriero/se_logs/kitchen/pa.npy", pa)
    np.save("/mnt/vast/home/randall.balestriero/se_logs/kitchen/official.npy", official)
    print("\nsaved fa/pa/official npy")


if __name__ == "__main__":
    main()
