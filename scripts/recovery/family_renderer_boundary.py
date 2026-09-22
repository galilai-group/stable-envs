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
    "dmc": ("DMCAcrobotSwingup", "DMCPendulumSwingup", "DMCCartpoleSwingup"),
    "ogbench": ("PointMazeLargeNavigate",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for name in COHORTS[args.cohort]:
        corpus_name = "PointmazeLargeNavigate" if name == "PointMazeLargeNavigate" else name
        path = Path("tests/fixtures/regression/join/step4/states") / f"{corpus_name}.npz"
        with np.load(path) as corpus:
            rows = corpus["states"][:4].copy()
        with se.make(name, num_envs=len(rows), device=args.device) as env:
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
