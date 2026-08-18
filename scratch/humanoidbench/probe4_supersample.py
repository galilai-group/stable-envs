"""Does supersampling close the floor residual, and what does the picture score if it does not?

L0 in probe3 puts 99.7 % of the interior error on the floor, and the floor is a PLANE MJWarp can UV.
Two hypotheses are separable by this sweep:

  * **coverage** -- MuJoCo renders with ``offsamples=4`` MSAA and the floor's ``mark="edge"`` grid
    lines are about one pixel wide, so a single ray per pixel lands on one side of a line where MuJoCo
    averages both. Supersampling FIXES that.
  * **minification** -- MuJoCo mipmaps and MJWarp point-samples the base level. Supersampling does not
    fix that (measured at 0.6-1.4 % exact on the locomotion suite), and it blurs everything else.

The two are told apart by the near band and the far band moving in different directions.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, MjwRenderer, agreement, cam_id, jsave, make_env, mj_render,
                       mj_segment, renderer, resolved_model, sample_qpos, set_state)
from probe3_mjwarp import attribute, bands
from stable_envs.testing.conformance import ssim

SIZE = 256
NSTATE = 16


def downsample(a, ss):
    n, h, w, c = a.shape
    return (a.astype(np.float32).reshape(n, h // ss, ss, w // ss, ss, c).mean((2, 4))
            .round().clip(0, 255).astype(np.uint8))


def main():
    task = sys.argv[1]
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    import mujoco
    res = {"task": task, "robot": robot, "size": SIZE, "nstate": NSTATE}

    env, u = make_env(task, robot, SIZE)
    cam_name, path = u.camera_name, u.fullpath
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    env.close()

    m, _ = resolved_model(path, SIZE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    ref = mj_render(m, d, cam, SIZE, Q, tag="ref")
    seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")

    sweep = {}
    for ss in (1, 2, 3, 4):
        R = MjwRenderer(m, cam, SIZE * ss, NSTATE)
        got = R.render(Q)
        if ss > 1:
            got = downsample(got, ss)
        a = agreement(got, ref)
        a["ssim"] = round(ssim(got, ref), 6)
        a["bands"] = bands(got, ref, seg, m)
        a["by_class"] = attribute(got, ref, seg, m)
        sweep[f"ss{ss}"] = a
        del R
    res["supersample"] = sweep

    # per-state spread at the chosen setting, which is what a threshold has to clear
    R = MjwRenderer(m, cam, SIZE, NSTATE)
    got = R.render(Q)
    per = [agreement(got[i:i + 1], ref[i:i + 1]) for i in range(NSTATE)]
    res["per_state_ss1"] = dict(
        interior_exact_min=round(min(p["interior_exact"] for p in per), 6),
        interior_exact_max=round(max(p["interior_exact"] for p in per), 6),
        mean_abs_min=round(min(p["mean_abs"] for p in per), 4),
        mean_abs_max=round(max(p["mean_abs"] for p in per), 4),
        within8_min=round(min(p["within8_frac"] for p in per), 6),
        within8_max=round(max(p["within8_frac"] for p in per), 6),
        ssim_min=round(min(ssim(got[i:i + 1], ref[i:i + 1]) for i in range(NSTATE)), 6))
    del R

    # MuJoCo render-only throughput (no mj_forward), which is the fair denominator for a renderer
    r = renderer(m, SIZE, "ref")
    set_state(m, d, Q[0])
    r.update_scene(d, camera=int(cam))
    r.render()
    t0 = time.perf_counter()
    for _ in range(120):
        r.update_scene(d, camera=int(cam))
        r.render()
    render_only = round(120 / (time.perf_counter() - t0), 1)
    t0 = time.perf_counter()
    for i in range(60):
        set_state(m, d, Q[i % NSTATE])
        r.update_scene(d, camera=int(cam))
        r.render()
    end_to_end = round(60 / (time.perf_counter() - t0), 1)
    res["mujoco_fps"] = dict(render_only=render_only, restore_and_render=end_to_end)

    jsave(f"{LOGS}/probe4_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
