"""Is MJWarp bit-deterministic on these scenes? Stress form, one cell per process.

``test_determinism`` renders the same states twice and demands identical bytes. It failed on
``h1hand-powerlift`` in one run and on ``h1hand-slide`` in the next, i.e. on a DIFFERENT cell each
time, which is the signature of a rare nondeterminism rather than of a scene that is always wrong.
Two renders is not enough to see that, so this does 40 and counts.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
REPS = 40


def main():
    task, robot = sys.argv[1], sys.argv[2]
    import stable_envs as se
    name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
            f"{''.join(p.capitalize() for p in task.split('_'))}")
    env = se.make(name, batch=8, device="cuda")
    st = env.sample_states(env.B)
    base = env.render(st).cpu().numpy()
    worst, bad, px = 0, 0, 0
    for _ in range(REPS):
        f = env.render(st).cpu().numpy()
        d = np.abs(f.astype(np.int16) - base.astype(np.int16))
        if d.max():
            bad += 1
            worst = max(worst, int(d.max()))
            px = max(px, int((d.max(-1) > 0).sum()))
    out = {"class": name, "reps": REPS, "differing_renders": bad, "max_abs": worst,
           "max_differing_px": px}
    with open(f"{LOGS}/det_{robot}_{task}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)


if __name__ == "__main__":
    main()
