"""Does MJWarp's ``track`` camera move without ``com_pos``? One env per process, honest ordering.

``mujoco_warp.kinematics`` writes xpos/xquat/xmat/xipos and NOT ``subtree_com``; a ``trackcom``
camera's position is ``subtree_com[body] + cam_poscom0``. So the four-call frame loop the repo uses
(kinematics -> camlight -> refit_bvh -> render) should leave the camera wherever ``put_data``'s
initial ``mj_forward`` put it.

The first version of this check ran ``com_pos=True`` first and ``com_pos=False`` second on the SAME
renderer and the SAME states, and reported agreement -- because the second pass inherited the first
pass's ``subtree_com``. Order matters, so each arm gets its own renderer and its own process.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import ENVS, agreement, jsave, make_env, mj_render, sample_qpos, track_cam

SIZE = 240
N = 8


def run(env_id, com):
    import mujoco

    from mjw_common import MjwRenderer
    env, u = make_env(env_id, SIZE)
    m, d = u.model, u.data
    cam = track_cam(m)
    st = sample_qpos(env_id, N, seed=0)
    old = int(m.opt.solver)
    m.opt.solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    try:
        W = MjwRenderer(m, cam, SIZE, N)
    finally:
        m.opt.solver = old
    got = W.render(st, com=com)                        # the ONLY render this process performs
    warp_cam = W.cam_xpos()[:N, cam]
    ref_cam = []
    for q in st:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        ref_cam.append(np.array(d.cam_xpos[cam], np.float64))
    ref_cam = np.stack(ref_cam)
    ref = mj_render(m, d, cam, SIZE, st)
    return {"env": env_id, "com_pos": bool(com),
            "cam_xpos_max_err": round(float(np.abs(warp_cam - ref_cam).max()), 6),
            "cam_xpos_spread_warp": round(float(np.abs(warp_cam - warp_cam[0]).max()), 6),
            "cam_xpos_spread_ref": round(float(np.abs(ref_cam - ref_cam[0]).max()), 6),
            "parity": agreement(got, ref)}


if __name__ == "__main__":
    env_id, com = sys.argv[1], sys.argv[2] == "1"
    r = run(env_id, com)
    print(r, flush=True)
    jsave(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/"
          f"probe3_compos_{env_id.split('-')[0]}_{int(com)}.json", r)
