"""Fresh evidence for Rocket's atomic invalid-action normalization and valid domain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as md
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle
from tests.helpers.reference.rocket_semantics import capture, differences

from stable_envs._adapters.rocket_spaces import nodes
from stable_envs._adapters.rocket_task import RocketLandingEnv as Candidate


def snapshot(env):
    fields = capture(env)
    rng = {
        p: copy.deepcopy(n._np_random.bit_generator.state) if n._np_random is not None else None
        for p, n in nodes(env.variation_space)
    }
    return fields, rng, env.render().copy()


def equal(a, b):
    assert not differences(a[0], b[0]), differences(a[0], b[0])
    assert a[1] == b[1]
    assert np.array_equal(a[2], b[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    blocker = None
    for seed in [0, 17, 42]:
        for dtype in [np.float32, np.float64]:
            envs = [
                Oracle(render_mode="rgb_array", render_resolution=(64, 64)),
                Candidate(render_mode="rgb_array", render_resolution=(64, 64)),
                Candidate(render_mode="rgb_array", render_resolution=(64, 64)),
            ]
            oracle, candidate, clean = envs
            try:
                for e in envs:
                    e.reset(seed=seed)
                before = snapshot(candidate)
                invalid = []
                valid = []
                for dim in range(7):
                    lo, hi = (0, 1) if dim in [3, 4] else (-1, 1)
                    for bound in [lo, hi]:
                        for direction in [-np.inf, None, np.inf]:
                            value = dtype(bound) if direction is None else np.nextafter(dtype(bound), dtype(direction))
                            action = np.zeros(7, dtype=dtype)
                            action[dim] = value
                            (valid if lo <= value <= hi else invalid).append(action)
                # All 128 corners cover simultaneous fin mixing, ignition, throttle and gimbals.
                valid += [
                    np.array(x, dtype=dtype)
                    for x in itertools.product(*[([0, 1] if i in [3, 4] else [-1, 1]) for i in range(7)])
                ]
                invalid += [np.zeros(s, dtype=dtype) for s in [(), (6,), (8,), (1, 7), (7, 1)]]
                invalid += [np.zeros(7, dtype=t) for t in [np.int32, np.int64, np.float16, np.bool_, np.complex128]]
                for value in [np.nan, np.inf, -np.inf]:
                    for dim in range(7):
                        action = np.zeros(7, dtype=dtype)
                        action[dim] = value
                        invalid.append(action)
                # Preserve the old equal-count cancellation controls; neither path may mutate.
                for fin in [0.5, -0.4583333333333333, 0]:
                    invalid.append(np.array([fin, 0, 0, 0, 0, 2, 0], dtype=dtype))
                exceptions = {}
                for action in invalid * 3:
                    try:
                        candidate.step(action)
                    except (TypeError, ValueError, AssertionError) as error:
                        exceptions[type(error).__name__] = exceptions.get(type(error).__name__, 0) + 1
                    else:
                        raise AssertionError("Invalid action accepted")
                    equal(before, snapshot(candidate))
                suffix = np.array([0.2, -0.1, 0.3, 1, 0.5, 0.1, -0.2], dtype=dtype)
                for _ in range(3):
                    candidate.step(suffix.copy())
                    clean.step(suffix.copy())
                    equal(snapshot(candidate), snapshot(clean))
                # Reset per control: every boundary/corner must actually reach the physics path.
                for index, action in enumerate(valid):
                    for e in [oracle, candidate]:
                        e.reset(seed=seed)
                    try:
                        oracle.step(action.copy())
                        candidate.step(action.copy())
                    except BaseException as error:
                        blocker = dict(
                            seed=seed,
                            dtype=str(np.dtype(dtype)),
                            action=action.tolist(),
                            index=index,
                            exception=repr(error),
                            reason="supported action raises",
                        )
                        break
                    equal(snapshot(oracle), snapshot(candidate))
                records.append(
                    dict(
                        seed=seed,
                        dtype=str(np.dtype(dtype)),
                        valid_controls=len(valid),
                        invalid_calls=len(invalid) * 3,
                        exceptions=exceptions,
                        immediate_numeric_rng_pixels_exact=True,
                        next_three_valid_steps_exact=True,
                    )
                )
                print("boundary", seed, np.dtype(dtype), "blocker", blocker, flush=True)
            finally:
                for e in envs:
                    e.close()
            if blocker:
                break
        if blocker:
            break
    paths = [
        ROOT / p
        for p in [
            "tests/helpers/reference/rocket_semantics.py",
            "stable_envs/_adapters/rocket_actions.py",
            "stable_envs/_adapters/rocket_task.py",
            "stable_envs/_adapters/rocket_spaces.py",
            "scripts/research_rocket_production/action_boundary_gate.py",
        ]
    ]
    result = dict(
        status="SUPPORTED_ACTION_BLOCKER" if blocker else "ATOMIC_BOUNDARY_PASS",
        records=records,
        blocker=blocker,
        versions={k: md.version(k) for k in ["PyFlyt", "pybullet", "numpy", "gymnasium"]},
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope="Private migrated task boundary; public State/adapter acceptance still required",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert blocker is None, blocker


if __name__ == "__main__":
    main()
