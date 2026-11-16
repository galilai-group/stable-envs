"""Play the same start State and the same actions in stable-envs and stable-worldmodel (swm); write the state error.

    python -m benchmarks.rollout_parity --env PushT TwoRoom DMCReacher OGBCube --steps 20 --out rollout_parity.csv

Modes:
  reset  (default) both sides reset with the same seed. The script checks that the start States are equal, then
         both take the same random actions.
  set    ours resets with a seed, ``SwmSource.set_state`` writes that start State into swm, and both continue. swm's
         PushT setter takes one physics step after it writes the State, so in this mode PushT starts one step apart.

After every action the script compares the state vectors of ``swm_source.state_vector`` (positions and angle for
PushT, the agent position for TwoRoom, qpos and qvel for MuJoCo envs). The CSV has the start error, the largest
absolute error, and the error after the last action.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from benchmarks.speed import SEED, host_info

COLUMNS = [
    "env",
    "mode",
    "episodes",
    "steps",
    "start_abs_error",
    "max_abs_error",
    "final_abs_error",
    "first_step_over_1e-3",
    "source",
    "status",
    "host",
    "stable_envs_sha",
    "source_versions",
]


def measure(name: str, episodes: int, steps: int, mode: str = "reset") -> dict:
    import stable_envs as snv
    from benchmarks.swm_source import SwmSource, rows, source_id, state_vector, versions

    src = SwmSource(name)
    rng = np.random.default_rng(SEED)
    errors, start = np.zeros((episodes, steps)), np.zeros(episodes)
    with snv.make(name, num_envs=1, device="cpu") as env:
        for ep in range(episodes):
            env.reset(seed=SEED + ep)
            src.reset(seed=SEED + ep)
            s0 = rows(env.get_state())[0]
            if mode == "set":
                src.set_state(s0)
            start[ep] = float(np.abs(state_vector(s0) - src.get_state(s0)).max())
            for t in range(steps):
                action = env.action_spec.sample(1, rng)
                env.step(action, frames=None)
                src.step(np.asarray(action.cpu() if hasattr(action, "cpu") else action)[0])
                ours = rows(env.get_state())[0]
                theirs = src.get_state(ours)
                errors[ep, t] = float(np.abs(state_vector(ours) - theirs).max())
    src.close()
    over = np.flatnonzero((errors > 1e-3).any(0))
    return {
        "mode": mode,
        "episodes": episodes,
        "steps": steps,
        "start_abs_error": f"{start.max():.4g}",
        "max_abs_error": f"{errors.max():.4g}",
        "final_abs_error": f"{errors[:, -1].max():.4g}",
        "first_step_over_1e-3": int(over[0]) + 1 if len(over) else "",
        "source": source_id(name),
        "status": "ok",
        "source_versions": versions(),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--mode", choices=["reset", "set"], default="reset", help="how swm gets the start State")
    p.add_argument("--out", default="rollout_parity.csv")
    a = p.parse_args(argv)
    host = host_info(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name in a.env:
            try:
                row = measure(name, a.episodes, a.steps, a.mode)
            except Exception as e:  # noqa: BLE001
                row = {"status": f"error: {type(e).__name__}: {e}"[:300]}
            row.update(env=name, mode=a.mode, **host)
            print(row, flush=True)
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
