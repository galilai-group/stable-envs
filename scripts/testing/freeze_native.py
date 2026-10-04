"""Generate five independent source trajectories into a NEW sealed corpus.

The action plans match the historical native corpora, making migration equivalence
measurable. Source calls alone produce the expected states and task outputs.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from stable_envs.testing.corpus import write_trajectory
from stable_envs.testing.evidence import EvidenceRun, sha256
from stable_envs.testing.provenance import pinned_source_identity
from tests.helpers.reference.bridges import NativeBridge, load_source
from tests.helpers.reference.profiles import PINS, PROFILES
from tests.helpers.reference.saved import SWM_PIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    identity = pinned_source_identity(args.source, SWM_PIN, PINS["native"])
    load_source(args.source, "native")
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / "stable_envs/testing/corpus.py"]
    files += list((root / "tests/helpers/reference").glob("*.py"))
    producer = dict(
        commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        tracked_dirty=bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
            )
        ),
        files={str(path.relative_to(root)): sha256(path) for path in files},
    )
    run = EvidenceRun(args.output)
    required = [case.registration for case in PROFILES["native"]]
    run.json("plan.json", dict(required=required, scope="State/task trajectories; no render or acceptance claim"))
    for case in PROFILES["native"]:
        bridge = NativeBridge(case, 3)
        try:
            bridge.reset(17)
            initial = bridge.source.get_state()
            rng = np.random.default_rng(217)
            if case.dtypes == ("int64",):
                actions = rng.integers(0, 2 if case.domain == "cartpole" else 3, (32, 3, 1), dtype=np.int64)
            else:
                bound = 2 if case.domain == "pendulum" else 1
                actions = rng.uniform(-bound, bound, (32, 3, 1)).astype(np.float32)
            outcomes = [bridge.step(action) for action in actions]
            values = {"actions": actions}
            for key in initial.keys(True, True):
                name = key if isinstance(key, str) else "/".join(key)
                values["state/" + name] = np.stack(
                    [initial[key].numpy(), *[outcome["state"][key].numpy() for outcome in outcomes]]
                )
            for key in ("reward", "terminated", "truncated", "observation", "info_state"):
                values[key] = np.stack([outcome[key].numpy() for outcome in outcomes])
            write_trajectory(
                run.root / case.domain,
                registration=case.registration,
                seed=17,
                source=identity,
                producer=producer,
                arrays=values,
            )
        finally:
            bridge.close()
    run.json("result.json", dict(status="GENERATED", registrations=required, steps=32, batch=3))
    run.seal()
    print(run.root)


if __name__ == "__main__":
    main()
