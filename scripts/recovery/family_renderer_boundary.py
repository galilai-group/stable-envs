"""Compare frozen renderer rows for the first bounded family recovery cohort.

Run this identical driver against baseline and candidate in separate processes.
The output is a frame digest, not a semantic oracle or an upstream similarity claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import stable_envs as se

COHORTS = {
    "dmc": ("DMCAcrobotBalance", "DMCPendulumSwingup", "DMCCartpoleSwingup"),
    "dmc_continuation": ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk"),
    "native_classic": ("CartPole", "Pendulum"),
    "native_classic_remaining": ("Acrobot", "MountainCar", "MountainCarContinuous"),
    "native_navigation": ("TwoRoom", "Piecewise", "SimplePointMaze"),
    "dmc_aligned": ("DMCBallInCupCatch", "DMCFingerTurnHard", "DMCManipulatorBringBall", "DMCQuadrupedWalk"),
    "ogbench": ("PointMazeLargeNavigate",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-baseline", action="store_true")
    args = parser.parse_args()
    results = {}
    for name in COHORTS[args.cohort]:
        corpus_name = {
            "PointMazeLargeNavigate": "PointmazeLargeNavigate",
            "DMCAcrobotBalance": "DMCAcrobotSwingup",
        }.get(name, name)
        lookup_name = "DMCAcrobotSwingup" if args.historical_baseline and name == "DMCAcrobotBalance" else name
        corpus_paths = {
            "DMCManipulatorBringBall": "step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz",
            "DMCQuadrupedWalk": "step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz",
        }
        path = Path("tests/fixtures/regression/join") / corpus_paths.get(name, f"step4/states/{corpus_name}.npz")
        with np.load(path) as corpus:
            rows = corpus["states"][:4].copy()
        with se.make(lookup_name, num_envs=len(rows), device=args.device) as env:
            render = getattr(env, "_render", env.render)
            frame = render(rows).cpu().numpy()
            results[name] = {
                "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "shape": list(frame.shape),
            }
        print(name, results[name], flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
