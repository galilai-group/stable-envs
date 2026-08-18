"""Measure what a HumanoidBench cell actually does, THROUGH THE ENV CLASS, on its own states.

Writes one JSON per cell; the ``FIDELITY`` table in ``stable_envs/envs/humanoidbench.py`` is filled
from these by hand, because a threshold a script writes into the file it then tests is not a threshold.

Four INDEPENDENT state draws, not one: the spread over draws is what says how much headroom the gate
needs, and a threshold set from a single draw has a margin that is a guess.

ONE CELL PER PROCESS. Two MuJoCo envs alive at once can return byte-identical frames from cameras that
provably differ, and a CUDA 700 poisons the process.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
N = 64
BATCH = 16


def main():
    task, robot = sys.argv[1], sys.argv[2]
    from stable_envs.envs import humanoidbench as HB
    from stable_envs.testing.conformance import pixel_agreement, ssim

    cname = next(c for c, r, t in HB.CELLS if r == robot and t == task)
    cls = type(cname, (HB.HumanoidBenchBase,), dict(ROBOT=robot, TASK=task))
    ref = cls.reference_env(cls.default_size)

    draws = []
    for s in range(4):
        env = cls(batch=BATCH, device="cuda", seed=s)
        st = env.sample_states(N)
        official = cls.reference_render(ref, st)
        ours = env.render(st).cpu().numpy()
        assert ours.shape == official.shape, f"{ours.shape} vs {official.shape}"
        a = pixel_agreement(ours, official)
        a["ssim"] = ssim(ours, official)
        draws.append({k: (float(v) if v is not None else None) for k, v in a.items()})
        del env

    keys = ("mean_abs", "within8_frac", "ssim", "interior_mean_abs", "interior_exact_frac",
            "exact_frac", "interior_frac")
    spread = {k: [min(d[k] for d in draws), max(d[k] for d in draws)] for k in keys}
    res = {"class": cname, "robot": robot, "task": task, "n_states": N, "n_draws": 4,
           "state_dim": int(cls(batch=1, device="cuda").state_dim),
           "draw0": draws[0], "spread": spread}

    # throughput at the env's OWN surface: states in, uint8 frames out, everything included
    for b in (1, 16, 64):
        e = cls(batch=b, device="cuda")
        s = torch.as_tensor(e.sample_states(b), device="cuda")
        e.render(s)
        torch.cuda.synchronize()
        reps = max(3, int(1500 / b))
        t0 = time.perf_counter()
        for _ in range(reps):
            e.render(s)
        torch.cuda.synchronize()
        res[f"fps_batch{b}"] = round(b * reps / (time.perf_counter() - t0), 1)
        del e

    with open(f"{LOGS}/calib_{robot}_{task}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(cname, json.dumps(res["draw0"]), json.dumps(res["spread"]))


if __name__ == "__main__":
    main()
