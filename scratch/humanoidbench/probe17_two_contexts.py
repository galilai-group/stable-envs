"""Does a SECOND MJWarp render context on the same model perturb the first?

1,140 consecutive renders across all 19 live cells in a fresh process are bit-identical
(``probe16_session_flake.py``), and the conformance suite still fails ``test_determinism`` on one cell
per run. The suite does one thing a render loop does not: ``check_resolution_consistency`` renders at
2S, which builds a second render context on the SAME model at a different ``cam_res``.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
REPS = 80


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "slide"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    two = not (len(sys.argv) > 3 and sys.argv[3] == "one")
    import stable_envs as se
    name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
            f"{''.join(p.capitalize() for p in task.split('_'))}")
    env = se.make(name, batch=8, device="cuda")
    st = env.sample_states(env.B)
    base = env.render(st).cpu().numpy()
    if two:
        env.render(st, size=2 * env.size)                       # the second context
    bad, worst = 0, 0
    for _ in range(REPS):
        f = env.render(st).cpu().numpy()
        d = int(np.abs(f.astype(np.int16) - base.astype(np.int16)).max())
        if d:
            bad += 1
            worst = max(worst, d)
    out = {"class": name, "second_context": two, "reps": REPS, "flaky": bad, "max_abs": worst}
    with open(f"{LOGS}/twoctx_{robot}_{task}_{'two' if two else 'one'}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)


if __name__ == "__main__":
    main()
