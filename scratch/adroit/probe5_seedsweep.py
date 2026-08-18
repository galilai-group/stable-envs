"""The most charitable convention: can ANY fixed camera reproduce the official Adroit renderer?

probe4 established that the official frame depends on how many actions preceded the first render.
That alone does not settle the family, because a port could DECLARE a convention -- "the reference
camera is the one gymnasium freezes at a reset state" -- and reproduce exactly that. This tests
whether such a convention is even well defined.

For each scene, N processes each build one env, reset with a DIFFERENT seed, render immediately (so
the camera is frozen at a reset state, the most favourable case), then restore the SAME canonical
state and render it. If the resulting frames are identical, "camera frozen at reset" is a constant of
the family and the port is viable with a declared convention. If they are not, no fixed camera
reproduces the reference and the state->frame map is ill-defined for this family.

One env per process, so gymnasium's unbound-GL-context aliasing cannot contaminate the comparison.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import itertools
import json
import subprocess
import sys

import numpy as np

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
SIZE = 480
TMP = "/mnt/vast/home/randall.balestriero/se_logs/adroit"
SEEDS = (0, 1, 2, 3, 7, 11)


def worker(scene, seed, out_path):
    import gymnasium, gymnasium_robotics, mujoco                                # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(REF_ID[scene], render_mode="rgb_array", width=SIZE, height=SIZE)
    u = env.unwrapped
    u.reset(seed=int(seed))
    u.render()                                   # camera frozen at THIS reset state

    st = np.load(os.path.join(TMP, f"canon_{scene}.npz"))
    u.model.body_pos[:] = st["body_pos"]
    u.model.body_quat[:] = st["body_quat"]
    u.model.site_pos[:] = st["site_pos"]
    u.data.qpos[:] = st["qpos"]
    u.data.qvel[:] = st["qvel"]
    mujoco.mj_forward(u.model, u.data)
    frame = np.ascontiguousarray(u.render())
    c = u.mujoco_renderer.viewer.cam
    np.savez(out_path, frame=frame, lookat=np.asarray(c.lookat, np.float64))


def drive():
    out = {}
    for scene in REF_ID:
        frames, lookats = {}, {}
        for s in SEEDS:
            p = os.path.join(TMP, f"seed_{scene}_{s}.npz")
            subprocess.run([sys.executable, __file__, scene, str(s), p], check=True,
                           env={**os.environ, "MUJOCO_GL": "egl"})
            z = np.load(p)
            frames[s], lookats[s] = z["frame"], z["lookat"]
        worst = {"pair": None, "differing_pixel_frac": 0.0, "mean_abs": 0.0, "max_abs": 0}
        n_identical = 0
        for a, b in itertools.combinations(SEEDS, 2):
            d = np.abs(frames[a].astype(np.int16) - frames[b].astype(np.int16)).max(-1)
            frac = float((d > 0).mean())
            if np.array_equal(frames[a], frames[b]):
                n_identical += 1
            if frac > worst["differing_pixel_frac"]:
                worst = {"pair": f"{a} vs {b}", "differing_pixel_frac": round(frac, 5),
                         "mean_abs": round(float(d.mean()), 3), "max_abs": int(d.max())}
        L = np.stack([lookats[s] for s in SEEDS])
        out[scene] = {
            "n_seed_pairs": len(list(itertools.combinations(SEEDS, 2))),
            "n_pairs_bit_identical": n_identical,
            "lookat_spread_m": np.ptp(L, axis=0).round(6).tolist(),
            "worst_pair": worst,
        }
        print(f"\n===== {scene} =====")
        print(json.dumps(out[scene], indent=2))
    with open(os.path.join(TMP, "probe5.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    if sys.argv[1] == "--drive":
        drive()
    else:
        worker(sys.argv[1], sys.argv[2], sys.argv[3])
