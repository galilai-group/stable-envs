"""Recheck saved Rocket pixels into a new sealed directory; never render or promote."""

import argparse
from pathlib import Path

import numpy as np

from stable_envs.testing.evidence import EvidenceRun
from tests.helpers.reference.saved import verify_rocket_render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = EvidenceRun(args.output)
    try:
        result = verify_rocket_render(args.input, Path(__file__).resolve().parents[2])
    except Exception as error:
        run.json("result.json", dict(status="FAIL", error=f"{type(error).__name__}: {error}"))
        try:
            with np.load(args.input / "frames.npz", allow_pickle=False) as archive:
                run.frame_diagnostic(
                    "worst-frame",
                    archive["ours"],
                    archive["oracle"],
                    metadata={
                        "input": str(args.input.resolve()),
                        "scope": "Largest pixel error, not necessarily the failing witness",
                    },
                )
        except (OSError, KeyError, ValueError, TypeError) as diagnostic_error:
            run.json("diagnostic-unavailable.json", dict(reason=str(diagnostic_error)))
        run.seal()
        raise
    run.json("result.json", result)
    run.seal()
    print(result)


if __name__ == "__main__":
    main()
