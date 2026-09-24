"""Preserve fused source reset versus State-derived symbolic outputs exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import stable_envs as se
from tests.helpers.reference.craftax_source import assert_engine, oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = oracle("CraftaxSymbolic")
    rows = []
    with se.make("CraftaxSymbolic", device="cpu") as env:
        for seed in [0, 17, 42]:
            expected, _ = source.reset(seed=seed)
            state = env.reset(seed=seed)
            assert_engine(source, env, state)
            supplied_key = jax.random.PRNGKey(seed)
            direct, _, _ = env._simulator.reset_fn(supplied_key)
            jax.block_until_ready(direct)
            np.testing.assert_array_equal(expected, np.asarray(direct))
            # Separate output JIT is used by the candidate projection; the source
            # reset's enclosing JIT can constant-fold intrinsic normalization.
            actual = np.asarray(env._simulator.output_fn(env._simulator.states[0]))
            source_standalone = np.asarray(jax.jit(source._env.get_obs, backend="cpu")(source._state))
            np.testing.assert_array_equal(actual, source_standalone)
            env.set_state(state)
            restored = np.asarray(env._simulator.output_fn(env._simulator.states[0]))
            np.testing.assert_array_equal(actual, restored)
            np.savez_compressed(
                args.output / f"reset-{seed}.npz",
                source_reset=expected,
                candidate_reset_return=np.asarray(direct),
                source_standalone=source_standalone,
                candidate_standalone=actual,
                restored=restored,
                **{"state/" + "/".join(k): v.cpu().numpy() for k, v in state.items(True, True)},
            )
            indices = np.flatnonzero(actual != expected)
            rows.append(
                {
                    "seed": seed,
                    "engine_state_and_keys_exact": True,
                    "direct_reset_return_exact": True,
                    "standalone_source_and_candidate_exact": True,
                    "restored_standalone_exact": True,
                    "historical_reset_output_recomputed_exact": len(indices) == 0,
                    "indices": indices.tolist(),
                    "source_values": expected[indices].tolist(),
                    "candidate_values": actual[indices].tolist(),
                    "maximum_difference": float(np.max(np.abs(actual - expected))),
                }
            )
    result = {
        "registration": "swm/CraftaxSymbolic-v1",
        "status": "BLOCKED_EXACT_RESET_OUTPUT",
        "cases": rows,
        "finding": "Exhaustive engine State and both keys reproduce dynamics, but standalone output JIT does not reproduce every bit of fused reset output. SWM retains _last_obs; candidate discards the returned output. Derived-output classification is insufficient for this historical reset boundary.",
        "scope": "CPU profile only. No tolerance relaxation, algorithm change, cached-observation schema addition, or promotion performed.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
