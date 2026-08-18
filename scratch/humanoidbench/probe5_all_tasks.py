"""L0 (the official scene, every correction applied) for ONE task. Driven as a SLURM array.

Also dumps a reference / ours / 4x-amplified-difference strip so the residual can be judged rather
than taken on the numbers.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, MjwRenderer, agreement, jsave, make_env, mj_render, mj_segment,
                       resolved_model, sample_qpos)
from probe3_mjwarp import attribute, bands
from stable_envs.testing.conformance import ssim

SIZE = 256
NSTATE = 16


def main():
    task, robot = sys.argv[1], sys.argv[2]
    dump = len(sys.argv) > 3 and sys.argv[3] == "dump"
    import mujoco
    res = {"task": task, "robot": robot, "size": SIZE, "nstate": NSTATE}
    try:
        env, u = make_env(task, robot, SIZE)
        cam_name, path = u.camera_name, u.fullpath
        res["camera"] = cam_name
        Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
        env.close()

        m, _ = resolved_model(path, SIZE)
        d = mujoco.MjData(m)
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        ref = mj_render(m, d, cam, SIZE, Q, tag="ref")
        seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")
        res["n_colors"] = int(len(np.unique(ref.reshape(-1, 3), axis=0)))

        R = MjwRenderer(m, cam, SIZE, NSTATE)
        res["texture_gaps"] = [g[1] for g in R.texture_gaps]
        res["prepared_hidden"] = len(R.prepared.hidden_geoms) if R.prepared else 0
        got = R.render(Q)
        a = agreement(got, ref)
        a["ssim"] = round(ssim(got, ref), 6)
        res["L0"] = a
        res["by_class"] = attribute(got, ref, seg, m)
        res["bands"] = bands(got, ref, seg, m)
        per = [agreement(got[i:i + 1], ref[i:i + 1]) for i in range(NSTATE)]
        res["per_state"] = dict(
            interior_exact_min=round(min(p["interior_exact"] for p in per), 6),
            mean_abs_max=round(max(p["mean_abs"] for p in per), 4),
            within8_min=round(min(p["within8_frac"] for p in per), 6),
            ssim_min=round(min(ssim(got[i:i + 1], ref[i:i + 1]) for i in range(NSTATE)), 6))
        if dump:
            import imageio.v2 as iio
            k = 0
            diff = np.clip(np.abs(got[k].astype(np.int16) - ref[k].astype(np.int16)) * 4,
                           0, 255).astype(np.uint8)
            strip = np.concatenate([ref[k], got[k], diff], axis=1)
            os.makedirs(f"{LOGS}/img", exist_ok=True)
            iio.imwrite(f"{LOGS}/img/{robot}_{task}.png", strip)
    except Exception as e:                                              # pragma: no cover
        res["error"] = f"{type(e).__name__}: {e}"
    jsave(f"{LOGS}/probe5_{robot}_{task}.json", res)
    print(res.get("error") or res["L0"])


if __name__ == "__main__":
    main()
