"""Source-only numeric before/after-render diagnostics on valid action prefixes."""

import argparse
from pathlib import Path

import numpy as np

from .bridges import Bridge
from .common import array_hash, catalog, profile, state_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = p.parse_args()
    records = []
    for registration in ["swm/FetchPickAndPlace-v3", "swm/FetchPickAndPlaceDense-v3", "swm/FetchPickAndPlaceDict-v3"]:
        for seed in [0, 17, 42]:
            for mode in ["zero", "random"]:
                for step in [0, 1, 10, 25, 49, 50]:
                    b = Bridge(catalog()[registration], seed, {})
                    try:
                        b.reset(seed)
                        actions = (
                            np.zeros((step, 1, 4), np.float64)
                            if mode == "zero"
                            else np.random.default_rng(90000 + seed).uniform(-1, 1, (step, 1, 4))
                        )
                        actions = actions.astype(args.dtype)
                        for action in actions:
                            b.step(action)
                        before = b.numeric_state()
                        b.render()
                        after = b.numeric_state()
                        changes = {
                            k: dict(
                                before=before[k].tolist(),
                                after=after[k].tolist(),
                                max_abs=float(np.max(np.abs(before[k] - after[k]))),
                            )
                            for k in before
                            if not np.array_equal(before[k], after[k])
                        }
                        records.append(
                            dict(
                                registration=registration,
                                seed=seed,
                                mode=mode,
                                action_dtype=args.dtype,
                                step=step,
                                prefix_hash=array_hash(actions),
                                before_hash=state_hash(before),
                                after_hash=state_hash(after),
                                changes=changes,
                            )
                        )
                    finally:
                        b.close()
                    write(args.output, dict(profile=profile(), records=records))


if __name__ == "__main__":
    main()
