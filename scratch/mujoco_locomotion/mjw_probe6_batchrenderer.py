"""End-to-end check of the ``com_pos`` fix through ``stable_envs.render.mjwarp.BatchRenderer``.

The class is CUDA-graph captured, so the conditional launch has to be inside the captured region --
if it were skipped at capture time and needed later, the graph would replay without it. Asserted by
rendering two clearly different qpos through the graph and requiring the camera to move.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import make_env, track_cam


def main(env_id="HalfCheetah-v5"):
    import mujoco
    import torch

    from stable_envs.render import mjwarp as MW
    from mjw_common import sample_qpos

    env, u = make_env(env_id, 128)
    m, d = u.model, u.data
    cam = track_cam(m)
    st = sample_qpos(env_id, 4, seed=0)
    ref = []
    for q in st:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        ref.append(np.array(d.cam_xpos[cam], np.float64))
    ref = np.stack(ref)

    solver = int(m.opt.solver)
    m.opt.solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    try:
        br = MW.BatchRenderer(m, batch=4, res=128, camera=cam)
    finally:
        m.opt.solver = solver
    assert br.needs_com, "BatchRenderer did not detect the TRACKCOM camera"
    br.render(torch.as_tensor(st, dtype=torch.float32, device="cuda"))
    import warp as wp
    got = wp.to_torch(br.d.cam_xpos).cpu().numpy()[:, cam]
    print("reference spread", round(float(np.abs(ref - ref[0]).max()), 6))
    print("BatchRenderer  spread", round(float(np.abs(got - got[0]).max()), 6))
    print("max |warp - mujoco|", float(np.abs(got - ref).max()))
    assert np.abs(got - ref).max() < 1e-5, "the graph-captured frame did not run com_pos"
    print("OK")


if __name__ == "__main__":
    main(*sys.argv[1:])
