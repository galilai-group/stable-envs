"""Does ``render/nvdiff.py``'s mipmapped sampler draw HumanoidBench's floor where MJWarp cannot?

MJWarp's entire residual on this family is the floor's 2 cm ``mark="edge"`` grid line under
minification (probe3/probe6). ``stable_envs/render/nvdiff.py`` exists for exactly that operation and
reports 0.66-0.81 far-band exactness on Gymnasium's locomotion floor against MJWarp's 0.0000-0.0001.
This measures it on THIS floor, on the same states, against the same reference.

Runs in ``~/venvs/senvdiff`` (nvdiffrast + mujoco 3.9), which has no ``humanoid_bench``; the state pool
is dumped from ``~/venvs/se-hb`` first (``--dump``) and the MJCF is read straight off disk.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
ASSETS = "/mnt/vast/home/randall.balestriero/src/humanoid-bench/humanoid_bench/assets"
SIZE = 256
NSTATE = 16
CAM = {"walk": "cam_default", "run": "cam_default", "stand": "cam_default",
       "stair": "cam_default", "slide": "cam_default", "balance_simple": "cam_default"}


def dump(task, robot):
    from hb_common import sample_qpos
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    np.save(f"{LOGS}/pool_{robot}_{task}.npy", Q)
    print("dumped", Q.shape)


def main():
    task, robot = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "dump":
        return dump(task, robot)
    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND
    from stable_envs.testing.conformance import pixel_agreement, ssim
    from hb_common import resolved_model

    Q = np.load(f"{LOGS}/pool_{robot}_{task}.npy")
    path = os.path.join(ASSETS, "envs", f"{robot}_pos_{task}.xml")
    m, _ = resolved_model(path, SIZE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM[task])

    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), SIZE)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), SIZE)
    r = mujoco.Renderer(m, height=SIZE, width=SIZE)
    ref = []
    for q in Q:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=int(cam))
        ref.append(np.ascontiguousarray(r.render()))
    ref = np.stack(ref)

    res = {"task": task, "robot": robot, "mujoco": mujoco.__version__}
    mjcam = mujoco.MjvCamera()
    mjcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mjcam.fixedcamid = int(cam)
    for mode in (True, False):
        R = ND.BatchRenderer(m, res=SIZE, device="cuda")
        R.resolve_from_mujoco(m, d, camera=mjcam)
        R.mipmap = mode
        got = R.render_qpos(torch.as_tensor(Q, dtype=torch.float32), camera_id=int(cam))
        got = got.cpu().numpy() if hasattr(got, "cpu") else np.asarray(got)
        a = pixel_agreement(got, ref)
        a["ssim"] = ssim(got, ref)
        res["mipmap" if mode else "no_mipmap"] = {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in a.items()}
        del R

    import json
    with open(f"{LOGS}/probe11_{robot}_{task}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
