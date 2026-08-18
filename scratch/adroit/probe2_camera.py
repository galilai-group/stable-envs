"""Is the Adroit official frame a function of the state? The camera says no -- by how much?

gymnasium's OffScreenViewer._init_camera does

    self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])      # for i in 0,1,2
    self.cam.distance  = self.model.stat.extent

and MujocoRenderer._set_cam_config then applies DEFAULT_CAMERA_CONFIG on top. Fetch overrides all
four parameters there, so its camera is a constant of the family. Every Adroit env overrides only
`distance` and `azimuth`, so `lookat` keeps the median of geom_xpos -- and the viewer is built
LAZILY, at the first render() call, and then frozen for the life of the env.

The consequence, if it is real, is that the official frame depends on the state the env happened to
be in when it was first rendered, which is not part of any state vector. This measures it three ways:

  A. how far the camera moves between construction histories;
  B. how many pixels that is worth, for the SAME state rendered through two such envs;
  C. whether the median is nevertheless stable across the reset distribution (39 geoms, most of them
     the hand, and a median is robust to a few moved bodies -- so this is a real possibility and the
     difference between a NO-GO and a declared constant).
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
SIZE = 480


def make(ref_id, render=True):
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    kw = dict(render_mode="rgb_array", width=SIZE, height=SIZE) if render else {}
    return gymnasium.make(ref_id, **kw)


def cam_of(u):
    c = u.mujoco_renderer.viewer.cam
    return dict(type=int(c.type), fixedcamid=int(c.fixedcamid),
                lookat=np.asarray(c.lookat, np.float64).copy(),
                distance=float(c.distance), azimuth=float(c.azimuth),
                elevation=float(c.elevation))


def main():
    out = {}
    for scene, ref in REF_ID.items():
        res = {}

        # ---- C: how stable is median(geom_xpos) over the reset distribution? -------------
        env0 = make(ref, render=False)
        u0 = env0.unwrapped
        env0.action_space.seed(0)
        meds_reset, meds_rolled = [], []
        for i in range(32):
            u0.reset(seed=i)
            meds_reset.append(np.median(u0.data.geom_xpos, axis=0).copy())
            for _ in range(int(np.random.default_rng(i).integers(1, 26))):
                u0.step(env0.action_space.sample())
            meds_rolled.append(np.median(u0.data.geom_xpos, axis=0).copy())
        meds_reset = np.stack(meds_reset)
        meds_rolled = np.stack(meds_rolled)
        res["median_geom_xpos_reset_spread_m"] = np.ptp(meds_reset, axis=0).round(6).tolist()
        res["median_geom_xpos_rolled_spread_m"] = np.ptp(meds_rolled, axis=0).round(6).tolist()
        res["median_geom_xpos_reset_vs_rolled_maxdiff_m"] = float(
            np.abs(meds_reset - meds_rolled).max())

        # ---- A: two envs, two construction histories, cameras compared ------------------
        # env A renders straight after reset(seed=0); env B takes 25 random actions first.
        envA = make(ref)
        uA = envA.unwrapped
        uA.reset(seed=0)
        uA.render()                                     # builds and FREEZES the viewer camera
        camA = cam_of(uA)

        envB = make(ref)
        uB = envB.unwrapped
        uB.reset(seed=0)
        envB.action_space.seed(0)
        for _ in range(25):
            uB.step(envB.action_space.sample())
        uB.render()                                     # same env, different first-render state
        camB = cam_of(uB)

        res["camA"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in camA.items()}
        res["camB"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in camB.items()}
        res["camera_lookat_shift_m"] = float(np.abs(camA["lookat"] - camB["lookat"]).max())

        # ---- B: the SAME state through both envs ----------------------------------------
        # Identical qpos/qvel AND identical per-episode model fields, so every input to the frame
        # except the frozen camera is equal by construction.
        import mujoco
        uA.reset(seed=7)
        qpos = np.array(uA.data.qpos, np.float64)
        qvel = np.array(uA.data.qvel, np.float64)
        body_pos = np.array(uA.model.body_pos, np.float64)
        body_quat = np.array(uA.model.body_quat, np.float64)
        site_pos = np.array(uA.model.site_pos, np.float64)

        frames = []
        for u in (uA, uB):
            u.model.body_pos[:] = body_pos
            u.model.body_quat[:] = body_quat
            u.model.site_pos[:] = site_pos
            u.data.qpos[:] = qpos
            u.data.qvel[:] = qvel
            mujoco.mj_forward(u.model, u.data)
            frames.append(np.ascontiguousarray(u.render()))
        d = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).max(-1)
        res["same_state_two_histories"] = {
            "max_abs_diff": int(d.max()), "differing_pixel_frac": float((d > 0).mean()),
            "mean_abs": float(d.mean())}
        out[scene] = res
        print(f"\n===== {scene} =====")
        print(json.dumps(res, indent=2, default=str))
        # envs are NOT closed: releasing an EGL context corrupts every other live one.

    with open("/mnt/vast/home/randall.balestriero/se_logs/adroit/probe2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
