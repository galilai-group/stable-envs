"""Separate presentation, upstream reset-key policy and shared dynamics experimentally."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import jax
import numpy as np

from stable_envs._adapters.craftax import numeric_leaves
from tests.helpers.reference.craftax_source import oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for classic in [True, False]:
        prefix = "Craftax" + ("Classic" if classic else "")
        pixel, symbolic = oracle(prefix + "Pixels"), oracle(prefix + "Symbolic")
        definition = inspect.getsource(type(symbolic._env).reset_env)
        file = Path(inspect.getsourcefile(type(symbolic._env)))
        rows = []
        for seed in [0, 17, 42]:
            pixel.reset(seed=seed)
            symbolic.reset(seed=seed)
            left, right = dict(numeric_leaves(pixel._state)), dict(numeric_leaves(symbolic._state))
            same_seed_exact = all(np.array_equal(v, right[k]) for k, v in left.items())
            assert same_seed_exact == classic
            if not classic:
                # Pixel's outer split of this transformed key supplies exactly
                # Symbolic's extra reset_env split to the shared world generator.
                transformed = jax.random.split(jax.random.PRNGKey(seed))[1]
                _, equivalent, _ = pixel._reset_fn(transformed)
                for key, value in numeric_leaves(equivalent):
                    np.testing.assert_array_equal(value, right[key])
            # Both presentations can consume one immutable engine State. Give
            # their wrappers the same owned key for identical future step input.
            symbolic._state, symbolic._key = pixel._state, pixel._key
            for index, action in enumerate([None, 0, 5, 1, 2, 4, 11]):
                if action is not None:
                    px = pixel.step(action)
                    sy = symbolic.step(action)
                    assert px[1:4] == sy[1:4]
                left, right = dict(numeric_leaves(pixel._state)), dict(numeric_leaves(symbolic._state))
                for key, value in left.items():
                    np.testing.assert_array_equal(value, right[key])
                np.testing.assert_array_equal(pixel._key, symbolic._key)
                keys = np.asarray(pixel._key).copy()
                frame = np.asarray(jax.jit(pixel._env.get_obs, backend="cpu")(pixel._state))
                vector = np.asarray(jax.jit(symbolic._env.get_obs, backend="cpu")(pixel._state))
                np.testing.assert_array_equal(pixel._key, keys)
                if index == 0:
                    np.savez_compressed(
                        args.output / f"{prefix}-same-state-{seed}.npz",
                        pixels=frame,
                        symbolic=vector,
                        wrapper_key=keys,
                        **{"engine/" + "/".join(k): v for k, v in left.items()},
                    )
            rows.append(
                {
                    "seed": seed,
                    "same_seed_world_exact": same_seed_exact,
                    "shared_state_six_step_continuation_exact": True,
                }
            )
        records.append(
            {
                "engine": "Classic" if classic else "Full",
                "symbolic_reset_definition": definition,
                "source_path": str(file),
                "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "cases": rows,
            }
        )
    result = {
        "status": "CRAFTAX_COMPONENT_SOURCE_PROOF_PASS",
        "correction": "Two shared engines, four concrete configurations. Full Symbolic adds an upstream reset key split before generate_world; the initial audit's only-get_obs-differs statement was too broad. Delegation has preserved that difference from the first implementation.",
        "records": records,
        "production_change_required": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
