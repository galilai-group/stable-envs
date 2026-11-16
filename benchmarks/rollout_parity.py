"""Play the same actions in stable-envs and stable-worldmodel (swm) and write the state error per env.

    python -m benchmarks.rollout_parity --env PushT TwoRoom DMCReacher OGBCube --steps 20 --out rollout_parity.csv

The script never writes a state into swm. It has two modes:
  seeded   both sides reset with the same seed, and the script compares the start states. Then both take the
           same random actions. For OGBCube and OGBScene, swm's env is the seeded class of
           ``tests/helpers/reference/ogbench_manipulation_source.py``: swm's own reset, task and physics code, with
           the task and variation draws taken from the seed.
  natural  swm resets with a seed and takes ``--offset`` random actions. The script reads swm's state, writes it
           into ours with ``env.set_state``, and both take the same next actions. The ``not_read`` column lists
           the fields of ours that swm does not expose; they keep the values of ours' seeded reset. For PushT,
           pymunk's contact solver caches are among them: ours starts them cold, so steps with contacts can
           move apart. That is a limit of the comparison, not a failure.

The error after each action is the largest absolute difference over the fields that ``SwmSource.read`` reads
from swm (``read_from_swm``): body poses and velocities for PushT, agent and target positions for TwoRoom, the
MuJoCo state for DM Control, and the MuJoCo state, controller and outputs for OGBench Cube and Scene.
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
    "worst_field",
    "read_from_swm",
    "not_read",
    "source",
    "status",
    "host",
    "stable_envs_sha",
    "source_versions",
]


def _field_names(state) -> list[str]:
    from stable_envs import schema_of

    schema = schema_of(state)
    if schema is not None:
        return list(schema.names)
    return [".".join(k) if isinstance(k, tuple) else k for k in state.keys(True, True)]


def _load(env, fields: dict) -> list[str]:
    """Write swm's fields into ours' current State with ``env.set_state``; return the fields of ours not read."""
    import torch

    from stable_envs import get_field, set_field

    state = env.get_state()
    for name, value in fields.items():
        field = get_field(state, name)
        set_field(state, name, torch.as_tensor(value, dtype=field.dtype).reshape(field.shape))
    env.set_state(state)
    read = {n.replace("/", ".") for n in fields}
    return [n for n in _field_names(state) if n.replace("/", ".") not in read]


def _error(env, fields: dict) -> tuple[float, str]:
    """The largest absolute difference between ours and swm over ``fields``, and the field where it is."""
    from stable_envs import get_field

    state = env.get_state()
    worst, where = 0.0, ""
    for name, value in fields.items():
        ours = get_field(state, name)[0].detach().cpu().double().numpy().reshape(-1)
        d = float(np.nan_to_num(np.abs(ours - value.reshape(-1))).max()) if ours.size else 0.0
        if d > worst:
            worst, where = d, name
    return worst, where


def measure(name: str, episodes: int, steps: int, mode: str = "seeded", offset: int = 10) -> dict:
    import stable_envs as snv
    from benchmarks.swm_source import SEEDED, SwmSource, source_id, versions

    src = SwmSource(name, seeded=mode == "seeded")
    rng = np.random.default_rng(SEED)
    errors, start = np.zeros((episodes, steps)), np.zeros(episodes)
    worst: dict = {}
    not_read: list[str] = []
    with snv.make(name, num_envs=1, device="cpu") as env:

        def act():
            a = env.action_spec.sample(1, rng)
            return a, np.asarray(a.cpu() if hasattr(a, "cpu") else a)[0]

        for ep in range(episodes):
            env.reset(seed=SEED + ep)
            src.reset(seed=SEED + ep)
            if mode == "natural":
                for _ in range(offset):
                    src.step(act()[1])
                not_read = _load(env, src.read())
            fields = src.read()
            start[ep] = _error(env, fields)[0]
            for t in range(steps):
                ours_action, swm_action = act()
                env.step(ours_action, frames=None)
                src.step(swm_action)
                errors[ep, t], where = _error(env, src.read())
                worst[where] = max(worst.get(where, 0.0), errors[ep, t])
    src.close()
    over = np.flatnonzero((errors > 1e-3).any(0))
    worst.pop("", None)
    return {
        "mode": mode,
        "episodes": episodes,
        "steps": steps,
        "start_abs_error": f"{start.max():.4g}",
        "max_abs_error": f"{errors.max():.4g}",
        "final_abs_error": f"{errors[:, -1].max():.4g}",
        "first_step_over_1e-3": int(over[0]) + 1 if len(over) else "",
        "worst_field": max(worst, key=worst.get) if worst else "",
        "read_from_swm": " ".join(fields),
        "not_read": " ".join(not_read),
        "source": source_id(name)
        + (f" ({type(src.env).__name__})" if mode == "seeded" and source_id(name) in SEEDED else ""),
        "status": "ok",
        "source_versions": versions(),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--mode", nargs="+", choices=["seeded", "natural"], default=["seeded", "natural"])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--offset", type=int, default=10, help="natural mode: swm steps before ours takes its state")
    p.add_argument("--out", default="rollout_parity.csv")
    a = p.parse_args(argv)
    host = host_info(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name in a.env:
            for mode in a.mode:
                try:
                    row = measure(name, a.episodes, a.steps, mode, a.offset)
                except Exception as e:  # noqa: BLE001
                    row = {"status": f"error: {type(e).__name__}: {e}"[:300]}
                row.update(env=name, mode=mode, **host)
                print({k: v for k, v in row.items() if k not in ("not_read", "source_versions")}, flush=True)
                w.writerow(row)
                f.flush()


if __name__ == "__main__":
    main()
