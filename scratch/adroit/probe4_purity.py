"""The state-sufficiency test for Adroit, run ONE ENV PER PROCESS.

probe3 showed the in-process two-env comparison cannot be trusted here: with two live viewers,
gymnasium's OffScreenViewer.render never calls make_context_current (it only does so when a single
MujocoRenderer owns more than one render_mode), so both renders resolve against whichever EGL context
was created last. Hammer/Pen/Relocate came back BYTE-IDENTICAL from cameras that differed by
0.013-0.045 m, which the same probe's single-viewer sensitivity sweep says must move 32-55% of pixels.

So the comparison is done across processes instead. Each process builds exactly one env, performs
`warmup` random actions BEFORE its first render (which is when gymnasium freezes the free camera's
lookat at median(geom_xpos)), then restores one canonical state and renders it. One EGL context per
process, nothing to alias.

    worker: python probe4_purity.py <scene> <warmup> <out.npy>
    driver: python probe4_purity.py --drive
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import subprocess
import sys

import numpy as np

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
SIZE = 480
TMP = "/mnt/vast/home/randall.balestriero/se_logs/adroit"


def canonical_state(u):
    """One fixed, fully-specified scene: qpos, qvel and every per-episode MODEL field."""
    import mujoco
    u.reset(seed=7)
    mujoco.mj_forward(u.model, u.data)
    return dict(qpos=np.array(u.data.qpos, np.float64), qvel=np.array(u.data.qvel, np.float64),
                body_pos=np.array(u.model.body_pos, np.float64),
                body_quat=np.array(u.model.body_quat, np.float64),
                site_pos=np.array(u.model.site_pos, np.float64))


def worker(scene, warmup, out_path):
    import gymnasium, gymnasium_robotics, mujoco                                # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(REF_ID[scene], render_mode="rgb_array", width=SIZE, height=SIZE)
    u = env.unwrapped
    u.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(int(warmup)):
        u.step(env.action_space.sample())
    u.render()                                   # <- the camera is frozen HERE

    st = np.load(os.path.join(TMP, f"canon_{scene}.npz"))
    u.model.body_pos[:] = st["body_pos"]
    u.model.body_quat[:] = st["body_quat"]
    u.model.site_pos[:] = st["site_pos"]
    u.data.qpos[:] = st["qpos"]
    u.data.qvel[:] = st["qvel"]
    mujoco.mj_forward(u.model, u.data)
    frame = np.ascontiguousarray(u.render())

    c = u.mujoco_renderer.viewer.cam
    np.savez(out_path, frame=frame, lookat=np.asarray(c.lookat, np.float64),
             distance=float(c.distance), azimuth=float(c.azimuth), elevation=float(c.elevation))
    # env deliberately not closed


def drive():
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    out = {}
    for scene, ref in REF_ID.items():
        env = gymnasium.make(ref)                       # render_mode=None: no viewer, no context
        st = canonical_state(env.unwrapped)
        np.savez(os.path.join(TMP, f"canon_{scene}.npz"), **st)

        paths = []
        for warmup in (0, 25):
            p = os.path.join(TMP, f"frame_{scene}_{warmup}.npz")
            subprocess.run([sys.executable, __file__, scene, str(warmup), p], check=True,
                           env={**os.environ, "MUJOCO_GL": "egl"})
            paths.append(p)
        a, b = (np.load(p) for p in paths)
        fa, fb = a["frame"], b["frame"]
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1)
        out[scene] = {
            "lookat_warmup0": a["lookat"].round(6).tolist(),
            "lookat_warmup25": b["lookat"].round(6).tolist(),
            "lookat_shift_m": float(np.abs(a["lookat"] - b["lookat"]).max()),
            "differing_pixel_frac": round(float((d > 0).mean()), 5),
            "mean_abs": round(float(d.mean()), 3), "max_abs": int(d.max()),
        }
        print(f"\n===== {scene} =====")
        print(json.dumps(out[scene], indent=2))
    with open(os.path.join(TMP, "probe4.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    if sys.argv[1] == "--drive":
        drive()
    else:
        worker(sys.argv[1], sys.argv[2], sys.argv[3])
