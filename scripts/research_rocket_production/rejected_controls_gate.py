"""Exact witness: rejected Rocket controls mutate hidden forces outside the step bound.

A successful execution confirms a blocker, not production acceptance. Uses only
official source/candidate public reset/step, and read-only numeric diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle
from tests.helpers.reference.rocket_semantics import capture, differences

from stable_envs._adapters.rocket_task import RocketLandingEnv as Candidate


def reject(env, action):
    try:
        env.step(action.copy())
    except AssertionError as error:
        return str(error)
    raise AssertionError("Official invalid control unexpectedly accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    witness = {}

    def run(seed, dtype, kind, equal_count):
        action = np.zeros(7, dtype=dtype)
        action[{"ignition": 3, "throttle": 4, "gimbal": 5}[kind]] = 2
        envs = [Oracle(), Oracle(), Candidate(), Candidate()]
        try:
            for e in envs:
                e.reset(seed=seed)
            if equal_count:
                alpha = (
                    envs[0].env.drones[0].lifting_surfaces.surfaces[0].physics_period
                    / envs[0].env.drones[0].lifting_surfaces.surfaces[0].cmd_tau
                )
                first = [action.copy() for _ in range(3)]
                second = [action.copy() for _ in range(3)]
                first[0][0] = 0.5
                first[1][0] = 0.5 * (alpha - 1)
            else:
                first = [action.copy()]
                second = [action.copy(), action.copy()]
            for e, prefix in zip(envs, [first, second, first, second], strict=True):
                for control in prefix:
                    reject(e, control)
            before = [capture(e) for e in envs]
            assert not differences(before[0], before[2]) and not differences(before[1], before[3])
            immediate = differences(before[0], before[1])
            assert not immediate, immediate
            assert all(e.step_count == 0 and e.env.physics_steps == 20 and e.env.aviary_steps == 10 for e in envs)
            for e in envs:
                e.step(np.zeros(7, dtype=dtype))
            after = [capture(e) for e in envs]
            assert not differences(after[0], after[2]) and not differences(after[1], after[3])
            future = differences(after[0], after[1])
            assert future and "wrapper/reward" in future
            rec = dict(
                seed=seed,
                dtype=dtype,
                invalid_control=kind,
                equal_rejected_call_count=equal_count,
                rejected_calls=[len(first), len(second)],
                last_action_equal=bool(np.array_equal(first[-1], second[-1])),
                prefixes=[np.asarray(first).tolist(), np.asarray(second).tolist()],
                immediate_91_numeric_fields_exact=True,
                public_step_count_before=0,
                physics_steps_before=20,
                aviary_steps_before=10,
                next_zero_action_differences=future,
                migrated_task_exact_to_source=True,
            )
            records.append(rec)
            if seed == 17 and dtype == "float64" and kind == "gimbal" and equal_count:
                witness.update({f"prefix_a/{i}": v for i, v in enumerate(first)})
                witness.update({f"prefix_b/{i}": v for i, v in enumerate(second)})
                for name, row in [
                    ("before_a", before[0]),
                    ("before_b", before[1]),
                    ("after_a", after[0]),
                    ("after_b", after[1]),
                ]:
                    witness.update({name + "/" + k: v for k, v in row.items()})
        finally:
            for e in envs:
                e.close()

    for seed in [0, 17, 42]:
        for dtype in ["float32", "float64"]:
            for kind in ["ignition", "throttle", "gimbal"]:
                run(seed, dtype, kind, False)
        for kind in ["ignition", "throttle", "gimbal"]:
            run(seed, "float64", kind, True)
    e = Oracle()
    try:
        e.reset(seed=17)
        bound = e.max_steps + 2
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
        for _ in range(4096):
            reject(e, action)
        horizon = dict(
            default_max_steps=e.max_steps,
            successful_physical_action_bound=bound,
            rejected_calls=4096,
            public_step_count=e.step_count,
            physics_steps=e.env.physics_steps,
            aviary_steps=e.env.aviary_steps,
            terminated=bool(e.termination),
            truncated=bool(e.truncation),
        )
        assert e.step_count == 0 and e.env.physics_steps == 20 and not (e.termination or e.truncation)
    finally:
        e.close()
    np.savez_compressed(a.output / "equal-count-witness.npz", **witness)
    paths = [
        ROOT / "tests/helpers/reference/rocket_semantics.py",
        Path(inspect.getfile(Oracle)),
        Path(inspect.getfile(Candidate)),
        ROOT / "stable_envs/_adapters/rocket_spaces.py",
    ]
    import PyFlyt

    base = Path(PyFlyt.__file__).parent
    paths += [
        base / p
        for p in [
            "core/aviary.py",
            "core/drones/rocket.py",
            "core/abstractions/boosters.py",
            "core/abstractions/gimbals.py",
            "core/abstractions/lifting_surfaces.py",
            "core/abstractions/boring_bodies.py",
            "gym_envs/rocket_envs/rocket_base_env.py",
        ]
    ]
    result = dict(
        status="REPLAY_BOUND_BLOCKER_REJECTED_CONTROLS",
        records=records,
        horizon=horizon,
        versions={k: md.version(k) for k in ["PyFlyt", "pybullet", "numpy", "gymnasium"]},
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        interpretation="The 1202 bound covers successful physics-advancing calls, not side-effecting rejected controls. Equal rejected-call count, identical final controls, and identical 91 named numeric diagnostics can hide different accumulated native forces. A rejected-call counter plus final caches is insufficient. This refutes the current bounded recipe, not every conceivable numeric restoration algorithm.",
        production_public_adapter_accepted=False,
        shared_contract_changes=False,
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output / "SHA256SUMS").write_text(
        "".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in sorted(a.output.iterdir()))
    )
    print(result["status"], len(records), flush=True)


if __name__ == "__main__":
    main()
