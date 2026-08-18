"""Probe 5: THE plate gate. How wrong is a baked static plate at pixels where no mover is drawn?

That quantity is the composite's error floor: wherever the segmentation says "not a mover" the
composite shows the plate, so any mover shadow, mover reflection or mover-driven lighting change that
lands on static geometry is an error no ray-caster can fix.

Measured directly with MuJoCo alone -- no mujoco_warp needed:
    sil(q)  = pixels whose segmentation object is a MOVER geom
    plate   = MuJoCo render with every mover hidden (state-independent by construction)
    error   = |full(q) - plate| restricted to ~sil(q)

Also reports the same restricted to INTERIOR pixels, which is the metric the suite gates on, and
splits the error by whether shadows are on.
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
from stable_envs.testing.conformance import interior_mask
from probe1_model import free_cam_pose

RES = 480
OUT = "/mnt/vast/home/randall.balestriero/se_logs/kitchen"


def build():
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    from gymnasium_robotics.envs.franka_kitchen import franka_env
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
    return m, mujoco.MjData(m), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")


def pool(n=24):
    """States the env actually reaches: the fixed reset qpos, plus random-action rollouts, plus the
    object joints driven over the range the tasks define."""
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make("FrankaKitchen-v1")
    env.action_space.seed(0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        env.reset(seed=int(i))
        for _ in range(int(rng.integers(0, 40))):
            env.step(env.action_space.sample())
        q = np.array(u.robot_env.data.qpos, np.float64)
        # drive each task's joints a random fraction of the way to its goal
        for task, idx in OBS_ELEMENT_INDICES.items():
            g = OBS_ELEMENT_GOALS[task]
            t = rng.uniform(0.0, 1.0)
            q[idx] = q[idx] * (1 - t) + g * t
        out.append(q)
    env.close()
    return np.stack(out)


def main():
    m, d, camid = build()
    r = mujoco.Renderer(m, width=RES, height=RES)
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vmovers = np.array([g for g in vis if g not in static], np.int64)
    print(f"visible geoms {len(vis)}, movers {len(vmovers)}")
    print(f"translucent visible: "
          f"{[(g, int(m.geom_bodyid[g]), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))) for g in vis if MW.effective_alpha(m,'geom')[g] < 1.0]}")

    def shot(q, hide=None, seg=False, shadows=True):
        keep = np.array(m.geom_group, copy=True)
        ks = int(m.vis.quality.shadowsize)
        if not shadows:
            m.vis.quality.shadowsize = 0
        if hide is not None:
            m.geom_group[hide] = 5
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        (r.enable_segmentation_rendering if seg else r.disable_segmentation_rendering)()
        r.update_scene(d, camera=camid)
        img = r.render().copy()
        r.disable_segmentation_rendering()
        m.geom_group[:] = keep
        m.vis.quality.shadowsize = ks
        return img

    states = pool(16)
    mover_set = set(vmovers.tolist())

    for shadows in (True, False):
        plate = shot(states[0], hide=vmovers, shadows=shadows)
        # confirm the plate is state-independent under this shadow setting
        p2 = shot(states[-1], hide=vmovers, shadows=shadows)
        dp = int(np.abs(plate.astype(np.int16) - p2.astype(np.int16)).max())

        errs, ierrs, sil_fracs, exact_out = [], [], [], []
        for q in states:
            full = shot(q, shadows=shadows)
            sg = shot(q, seg=True, shadows=shadows)
            oid, otype = sg[..., 0], sg[..., 1]
            sil = np.isin(oid, vmovers) & (otype == int(mujoco.mjtObj.mjOBJ_GEOM))
            # sites are drawn by MuJoCo and never by MJWarp; count them as movers for this probe so
            # the number isolates SHADOW/REFLECTION contamination rather than the known site gap
            sil |= (otype == int(mujoco.mjtObj.mjOBJ_SITE))
            outside = ~sil
            dd = np.abs(full.astype(np.int16) - plate.astype(np.int16)).max(-1)
            errs.append(float((dd[outside] == 0).mean()))
            exact_out.append(float(dd[outside].mean()))
            sil_fracs.append(float(sil.mean()))
            im = interior_mask(full[None])[0]
            oi = outside[:-1, :-1] & im
            ierrs.append(float((dd[:-1, :-1][oi] == 0).mean()) if oi.any() else float("nan"))
        print(f"\n== shadows={'ON' if shadows else 'OFF'} ==")
        print(f"  plate state-independent: max diff {dp}/255")
        print(f"  mover silhouette (incl. sites): {np.mean(sil_fracs):.4f} of the frame")
        print(f"  NON-mover pixels the plate gets EXACTLY right: {np.mean(errs):.6f} "
              f"(worst state {np.min(errs):.6f})")
        print(f"  NON-mover mean |diff|: {np.mean(exact_out):.4f}/255")
        print(f"  NON-mover INTERIOR pixels exact: {np.nanmean(ierrs):.6f} "
              f"(worst {np.nanmin(ierrs):.6f})")


if __name__ == "__main__":
    main()
