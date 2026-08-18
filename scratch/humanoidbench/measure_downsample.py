"""The per-env numbers the reproducibility suite needs, measured exactly as it measures them.

  * ``DOWNSAMPLE_EXACT_FRAC`` -- render at 2S, box-downsample, compare to S, on the states the test's
    own ``inputs`` fixture draws (``render_worker.sample_inputs``), so the recorded number is the one
    the test will see rather than a number from a different experiment.
  * ``check_batch_size_invariance`` over the test's own sweep (1, 3, 7, 8, 64, 1024) and determinism,
    both re-measured after the fixed-world-count change, because both failed the first run.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
BATCH_SIZES = (1, 3, 7, 8, 64, 1024)


def main():
    task, robot = sys.argv[1], sys.argv[2]
    import stable_envs as se
    from stable_envs.testing import conformance as C
    from stable_envs.testing import render_worker as W

    name = (f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
            f"{''.join(p.capitalize() for p in task.split('_'))}")
    env = se.make(name, batch=8, device="cuda")
    # NO style: ``test_resolution_downsample_agrees_within_the_measured_bound`` calls
    # ``check_resolution_consistency(env, states)`` with the DEFAULT appearance, and a number measured
    # under a randomised one is a number for a different experiment.
    states, _ = W.sample_inputs(env, 8)
    r = C.check_resolution_consistency(env, states)
    out = {"class": name, "downsample": r}

    st = env.sample_states(env.B)
    base = env.render(st).cpu().numpy()
    worst = 0
    for _ in range(3):
        again = env.render(st).cpu().numpy()
        worst = max(worst, int(np.abs(again.astype(np.int16) - base.astype(np.int16)).max()))
    out["determinism_max_abs"] = worst

    sweep = env.sample_states(BATCH_SIZES[-1], rng=np.random.default_rng(11))
    out["batch_size_invariance"] = C.check_batch_size_invariance(env, sweep, sizes=BATCH_SIZES)
    with open(f"{LOGS}/down_{robot}_{task}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f'    "{name}":{" " * max(1, 30 - len(name))}{r["exact_frac"]:.6f},   '
          f'# mean {r["mean_abs"]:.4f}/255   det {out["determinism_max_abs"]}   '
          f'binv {out["batch_size_invariance"]["max_abs_diff"]}')


if __name__ == "__main__":
    main()
