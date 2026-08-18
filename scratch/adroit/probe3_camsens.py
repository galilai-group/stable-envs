"""Isolate the camera term: how many pixels is a lookat shift worth, and do two envs really differ?

probe2 reported 80.9% differing pixels for Door and exactly 0.0 for Hammer/Pen/Relocate, from lookat
shifts of 0.216 / 0.020 / 0.013 / 0.045 m. A clean 0 alongside a nonzero shift is not a small effect,
it is a broken comparison, so this separates the two claims:

  1. SENSITIVITY, through ONE viewer: render one fixed state at lookat, lookat+d for a range of d.
     No second context, no second env, nothing to alias.
  2. IDENTITY, across two envs: re-read both cameras immediately before each comparison render, and
     report them next to the pixel difference, so "the cameras differed" is evidence rather than an
     assumption about when gymnasium freezes them.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
SIZE = 480


def make(ref_id):
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    return gymnasium.make(ref_id, render_mode="rgb_array", width=SIZE, height=SIZE)


def main():
    out = {}
    for scene, ref in REF_ID.items():
        res = {}
        env = make(ref)
        u = env.unwrapped
        u.reset(seed=7)
        u.render()                                          # build + freeze the viewer
        cam = u.mujoco_renderer.viewer.cam
        base_lookat = np.asarray(cam.lookat, np.float64).copy()

        mujoco.mj_forward(u.model, u.data)
        f0 = np.ascontiguousarray(u.render())

        # ---- 1. sensitivity through ONE viewer ------------------------------------------
        sens = {}
        for d in (0.001, 0.005, 0.013, 0.020, 0.045, 0.216):
            cam.lookat[:] = base_lookat + np.array([0.0, d, 0.0])
            f = np.ascontiguousarray(u.render())
            diff = np.abs(f.astype(np.int16) - f0.astype(np.int16)).max(-1)
            sens[f"dy={d}"] = {"differing_pixel_frac": round(float((diff > 0).mean()), 5),
                               "mean_abs": round(float(diff.mean()), 3),
                               "max_abs": int(diff.max())}
        cam.lookat[:] = base_lookat
        res["sensitivity_one_viewer"] = sens

        # ---- 2. identity across two envs ------------------------------------------------
        envB = make(ref)
        uB = envB.unwrapped
        uB.reset(seed=0)
        envB.action_space.seed(0)
        for _ in range(25):
            uB.step(envB.action_space.sample())
        uB.render()                                         # frozen at a DIFFERENT first state

        qpos = np.array(u.data.qpos, np.float64)
        qvel = np.array(u.data.qvel, np.float64)
        bp = np.array(u.model.body_pos, np.float64)
        bq = np.array(u.model.body_quat, np.float64)
        sp = np.array(u.model.site_pos, np.float64)

        frames, cams = [], []
        for uu in (u, uB):
            uu.model.body_pos[:] = bp
            uu.model.body_quat[:] = bq
            uu.model.site_pos[:] = sp
            uu.data.qpos[:] = qpos
            uu.data.qvel[:] = qvel
            mujoco.mj_forward(uu.model, uu.data)
            c = uu.mujoco_renderer.viewer.cam
            cams.append(dict(lookat=np.asarray(c.lookat, np.float64).round(6).tolist(),
                             distance=float(c.distance), azimuth=float(c.azimuth),
                             elevation=float(c.elevation)))
            frames.append(np.ascontiguousarray(uu.render()))
        d = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).max(-1)
        res["two_env_cameras_at_render_time"] = cams
        res["two_env_lookat_shift_m"] = float(
            np.abs(np.array(cams[0]["lookat"]) - np.array(cams[1]["lookat"])).max())
        res["two_env_frame_diff"] = {"differing_pixel_frac": round(float((d > 0).mean()), 5),
                                     "mean_abs": round(float(d.mean()), 3), "max_abs": int(d.max())}
        # are the two frames byte-identical objects, or genuinely equal pictures?
        res["frames_identical"] = bool(np.array_equal(frames[0], frames[1]))
        out[scene] = res
        print(f"\n===== {scene} =====")
        print(json.dumps(res, indent=2))

    with open("/mnt/vast/home/randall.balestriero/se_logs/adroit/probe3.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
