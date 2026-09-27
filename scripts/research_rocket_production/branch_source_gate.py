"""Independent source suffixes from real replay checkpoints; CPU official engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
from rocket_feasibility import RocketLandingEnv as Oracle
from rocket_feasibility import capture, differences

import stable_envs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = []
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.glob("stable_envs/_adapters/rocket*.py"))
        + [ROOT / "stable_envs/envs/rocket_semantics.py"]
    }
    for seed in [0, 17, 42]:
        for dtype in ["float32", "float64"]:
            for label, config, options, horizon in [
                ("reset", {}, {}, 0),
                ("mixed_prefix", {}, {}, 12),
                ("near_contact", {}, {}, 301),
                ("crash", {}, {}, 303),
                ("success", {"ceiling": 3.2}, {"accelerate_drop": False}, 7),
                ("horizon", {"ceiling": 500000.0}, {"accelerate_drop": False}, 1201),
            ]:
                config = dict(config, render_mode="rgb_array", render_resolution=(64, 64))
                zero = np.zeros(7, dtype=dtype)
                mixed = np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], dtype=dtype)
                prefix = [mixed if label == "mixed_prefix" else zero] * horizon
                e = stable_envs.make("PyFlytRocketLanding", device="cpu", simulator_config=config, seed=seed)
                try:
                    e.reset(seed=seed, options=options)
                    for action in prefix:
                        e.step(action[None])
                    state = e.get_state()
                    for branch, length in [("zero", 1), ("mixed", 10), ("alternating", 25)]:
                        oracle = Oracle(**config)
                        try:
                            oracle.reset(seed=seed, options=options)
                            for action in prefix:
                                oracle.step(action.copy())
                            # Destination has a different trajectory before each restore.
                            e.reset(seed=42, options={"variation": ["all"]})
                            for _ in range(5):
                                e.step(mixed[None])
                            e.set_state(state)
                            assert not differences(capture(oracle), capture(e._simulator.envs[0].env))
                            for i in range(length):
                                action = zero if branch == "zero" or branch == "alternating" and i % 2 == 0 else mixed
                                reference = oracle.step(action.copy())
                                actual = e.step(action[None])
                                assert actual["reward"][0].item() == reference[1]
                                assert actual["terminated"][0].item() == bool(reference[2])
                                assert actual["truncated"][0].item() == bool(reference[3])
                                assert not differences(capture(oracle), capture(e._simulator.envs[0].env))
                                assert np.array_equal(oracle.render(), e._simulator.envs[0].env.render())
                            records.append(
                                dict(
                                    seed=seed,
                                    dtype=dtype,
                                    checkpoint=label,
                                    horizon=horizon,
                                    branch=branch,
                                    suffix_steps=length,
                                    source_numeric_and_future_pixels_exact=True,
                                )
                            )
                        finally:
                            oracle.close()
                    print("SOURCE_SUFFIX", seed, dtype, label, "PASS", flush=True)
                finally:
                    e.close()
    (args.output / "result.json").write_text(
        json.dumps(
            dict(status="INDEPENDENT_SOURCE_SUFFIXES_PASS", records=records, production_hashes=source_hashes), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
