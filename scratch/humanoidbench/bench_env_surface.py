"""Throughput at the env's own surface AFTER the world count was pinned to 64.

The calibration run measured this while the context was still sized to the chunk, so batch 1 launched
one world. It now launches 64 whatever the batch, which is what makes ``check_batch_size_invariance``
structural, and the cost has to be measured rather than estimated.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"


def main():
    task, robot = sys.argv[1], sys.argv[2]
    import stable_envs as se
    name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
            f"{''.join(p.capitalize() for p in task.split('_'))}")
    out = {"class": name}
    for b in (1, 8, 16, 64, 256):
        env = se.make(name, batch=b, device="cuda")
        st = torch.as_tensor(env.sample_states(b), device="cuda")
        env.render(st)
        torch.cuda.synchronize()
        reps = max(3, int(1500 / b))
        t0 = time.perf_counter()
        for _ in range(reps):
            env.render(st)
        torch.cuda.synchronize()
        out[f"fps_batch{b}"] = round(b * reps / (time.perf_counter() - t0), 1)
        del env
    with open(f"{LOGS}/bench_{robot}_{task}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
