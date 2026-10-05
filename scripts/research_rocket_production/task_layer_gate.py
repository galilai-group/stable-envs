"""Bounded migrated-task parity before the rejected-control replay blocker."""

from __future__ import annotations

import argparse
import hashlib
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in [0, 17, 42]:
        for dtype in ["float32", "float64"]:
            for opts in [
                {},
                {"variation": ["all"]},
                {"variation": ["environment"]},
                {"variation": ["rocket.body_color", "pad.color"]},
            ]:
                source, target = Oracle(), Candidate()
                try:
                    source.reset(seed=seed, options=opts)
                    target.reset(seed=seed, options=opts)
                    assert not differences(capture(source), capture(target))
                    for _ in range(3):
                        action = np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], dtype=dtype)
                        source.step(action)
                        target.step(action)
                        assert not differences(capture(source), capture(target))
                    records.append(
                        dict(
                            seed=seed,
                            dtype=dtype,
                            options=opts,
                            reset_91_fields_exact=True,
                            three_step_91_fields_exact=True,
                        )
                    )
                finally:
                    source.close()
                    target.close()
                    if source.modified_pad_urdf:
                        Path(source.modified_pad_urdf).unlink(missing_ok=True)
    result = dict(
        status="PRIVATE_TASK_LAYER_PARITY_PASS",
        records=records,
        source_sha256=hashlib.sha256(Path(inspect.getfile(Oracle)).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(Path(inspect.getfile(Candidate)).read_bytes()).hexdigest(),
        scope="Private migration only; no public State/SimulatorAdapter/registration acceptance",
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output / "SHA256SUMS").write_text(
        hashlib.sha256((a.output / "result.json").read_bytes()).hexdigest() + "  result.json\n"
    )
    print(result["status"], len(records), flush=True)


if __name__ == "__main__":
    main()
