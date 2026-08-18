"""Characterise the rare 1-2/255 flake: does it need MANY envs alive in one process?

40 consecutive renders of every one of the 19 cells in a FRESH process are bit-identical
(``probe15_determinism.py``, 0/40 on all 19). Inside a 19-env pytest session, ``test_determinism`` and
``test_frame_i_is_independent_of_batch_size`` fail on ONE cell per run, and a different cell each run.
The variable that differs is how many MJWarp render contexts are alive, so this holds the number of
renders fixed and varies that.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
REPS = 60


def main():
    n_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    import stable_envs as se
    names = sorted(n for n, c in se.REGISTRY.items() if n.startswith("HumanoidBench"))[:n_envs]
    envs = {n: se.make(n, batch=8, device="cuda") for n in names}
    # touch every env once, so every context is built and live before anything is scored
    base = {}
    states = {}
    for n, e in envs.items():
        states[n] = e.sample_states(e.B)
        base[n] = e.render(states[n]).cpu().numpy()

    flakes, worst, total = {}, 0, 0
    for r in range(REPS):
        for n, e in envs.items():
            f = e.render(states[n]).cpu().numpy()
            total += 1
            d = int(np.abs(f.astype(np.int16) - base[n].astype(np.int16)).max())
            if d:
                flakes[n] = flakes.get(n, 0) + 1
                worst = max(worst, d)
    out = {"n_envs": len(names), "renders": total, "flaky_renders": sum(flakes.values()),
           "max_abs": worst, "per_env": flakes}
    with open(f"{LOGS}/flake_{len(names)}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)


if __name__ == "__main__":
    main()
