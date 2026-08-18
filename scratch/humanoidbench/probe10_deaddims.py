"""Does every ``qpos`` entry reach a pixel at 256 px? The dead-dimension gate, run before shipping.

``h1hand`` carries 50 finger joints on two Shadow hands seen from ~3 m through ``cam_default``. A
dimension the renderer provably ignores must not be declared, and ``check_state_dims_matter`` is the
test that would find it -- so it is measured here first, with the same donor-substitution rule.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import LOGS, MjwRenderer, jsave, make_env, resolved_model, sample_qpos

SIZE = 256
NSTATE = 8


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "walk"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    import mujoco
    env, u = make_env(task, robot, SIZE)
    cam_name, path = u.camera_name, u.fullpath
    names = [mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
             for j in range(u.model.njnt)]
    env.close()
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    D = sample_qpos(task, robot, NSTATE, seed=1, steps=60)          # donors
    m, _ = resolved_model(path, SIZE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R = MjwRenderer(m, cam, SIZE, NSTATE)
    base = R.render(Q)
    dead = []
    for k in range(int(m.nq)):
        pert = Q.copy()
        pert[:, k] = D[:, k]
        if np.array_equal(pert[:, k], Q[:, k]):
            continue
        if np.array_equal(R.render(pert), base):
            dead.append(k)
    res = {"task": task, "robot": robot, "nq": int(m.nq), "n_dead": len(dead), "dead": dead,
           "joint_names": names}
    jsave(f"{LOGS}/probe10_{robot}_{task}.json", res)
    print(f"{robot}-{task}: nq={m.nq} dead={len(dead)} {dead}")


if __name__ == "__main__":
    main()
