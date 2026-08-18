"""Why does RoboCasa's split disagree with the fused render? Three candidates, separated.

1. THE CAMERA MOVES. ``robot0_agentview_left`` and ``robot0_eye_in_hand`` are model cameras attached
   to a BODY, so their world pose is ``body_xpos + body_xmat @ cam_pos`` and depends on ``qpos``. A
   plate baked at the pinned pose is then a picture from the wrong place, and a plate cannot be keyed
   on the appearance alone.
2. THE MOVER SET IS TOO SMALL. Anything that moves and is not in it is baked into the plate at the
   pinned pose and shows as a ghost.
3. THE MOVER SET IS TOO BIG. Harmless for correctness, pure cost.

Reports the camera-pose spread over the pool, the symmetric difference between the structural mover
set and the one MEASURED over the pool, and -- for one state -- where the fused frame differs from the
plate while the dynamic pass reports no hit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import stable_envs as se

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="RoboCasaCounterToSink,RoboCasaCounterToSinkWrist")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()

    import mujoco

    from stable_envs.envs import robocasa as RC

    out = {}
    for name in a.envs.split(","):
        env = se.make(name, batch=8, size=a.size, device="cuda")
        cls = type(env)
        s = RC._scene(cls.KITCHEN)
        m = s.model
        d = mujoco.MjData(m)
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
        st = env.sample_states(a.n)

        cams, xps, xms = [], [], []
        for row in np.asarray(st, np.float64):
            q = s.pinned.copy()
            q[env._live] = row
            d.qpos[:] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            cams.append(np.array(d.cam_xpos[cid], np.float64))
            xps.append(np.array(d.geom_xpos, np.float64))
            xms.append(np.array(d.geom_xmat, np.float64))
        cams = np.stack(cams)
        xps, xms = np.stack(xps), np.stack(xms)
        moved = ((xps - xps[0]).any(0).any(-1) | (xms - xms[0]).any(0).any(-1))
        measured = {int(g) for g in np.flatnonzero(moved)
                    if int(m.geom_group[g]) in RC.VISIBLE_GROUPS}
        structural = {int(g) for g in RC._mover_geoms(cls.KITCHEN, cls.CAMERA, a.size)}
        missing = sorted(measured - structural)
        row = {
            "camera": cls.CAMERA,
            "cam_body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[cid])),
            "cam_mode": int(m.cam_mode[cid]),
            "cam_xpos_spread_m": float(np.abs(cams - cams[0]).max()),
            "n_structural": len(structural), "n_measured": len(measured),
            "missing_from_structural": [
                (g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in missing[:12]],
            "n_missing": len(missing),
            "extra_in_structural": len(structural - measured),
        }
        out[name] = row
        print(f"{name}: {json.dumps(row)}", flush=True)
        del env
        RC._CTX_CACHE.clear()
        RC._PLATE_CACHE.clear()
        torch.cuda.empty_cache()
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
