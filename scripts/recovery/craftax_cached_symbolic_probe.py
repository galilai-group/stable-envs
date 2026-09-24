"""Source-only replay proof for the measured SWM cached symbolic output leaf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.craftax_source import assert_engine, oracle, restore_oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with se.make("CraftaxSymbolic", device="cpu") as env:
        for seed in [0, 17, 42]:
            a, b = oracle("CraftaxSymbolic"), oracle("CraftaxSymbolic")
            expected, _ = a.reset(seed=seed)
            snapshot = env.reset(seed=seed)
            assert_engine(a, env, snapshot)
            snapshot["source_cache", "last_symbolic"] = torch.from_numpy(expected.copy())[None]
            saved = snapshot.clone()
            b.reset(seed=999)
            b.step(5)
            restore_oracle(b, env, snapshot)
            b._last_obs = snapshot["source_cache", "last_symbolic"][0].numpy().copy()
            np.testing.assert_array_equal(a._last_obs, b._last_obs)
            np.savez_compressed(
                args.output / f"reset-{seed}.npz",
                **{"state/" + "/".join(k): v.numpy() for k, v in snapshot.items(True, True)},
                source=a._last_obs,
                restored=b._last_obs,
            )
            snapshot["source_cache", "last_symbolic"].zero_()
            np.testing.assert_array_equal(saved["source_cache", "last_symbolic"][0].numpy(), b._last_obs)
            for action in [0, 5, 1, 2, 4, 11]:
                left, right = a.step(action), b.step(action)
                np.testing.assert_array_equal(left[0], right[0])
                assert left[1:4] == right[1:4]
            left, _ = a.reset()
            right, _ = b.reset()
            np.testing.assert_array_equal(left, right)
            rows.append(
                {"seed": seed, "cached_reset_exact": True, "snapshot_owning": True, "continuation_exact": True}
            )
    result = {
        "status": "SOURCE_CACHED_SYMBOLIC_REPLAY_PASS",
        "proposed_leaf": {"key": ["source_cache", "last_symbolic"], "shape": [8268], "dtype": "torch.float32"},
        "classification": "Numeric SWM-owned last-output cache required for exact historical reset output; not a dynamics substitute or a new Observation abstraction.",
        "validation": "Exact shape/dtype and finite values. This historical cache is independently restorable like last reward; no recomputation or approximate-equality validation may replace its exact retained values.",
        "registration": "swm/CraftaxSymbolic-v1",
        "cases": rows,
        "scope": "Only full symbolic demonstrated this necessity. No sibling schema or shared contract change is proposed.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
