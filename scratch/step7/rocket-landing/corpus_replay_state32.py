"""The state-quantisation term of the corpus replay, taken on its own: ``reference_render(state13)`` vs
the frozen frames (PyBullet's own renderer on the float32-rounded state; the frames were recorded from the
float64 state). CPU-only PyFlyt; provenance recorded (review D2 / acceptance rule for job 6395707).

Writes examples/results/step7/rocket-landing/tier/corpus_replay_state_quantisation_<host>.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import stable_envs as se  # noqa: E402
from stable_envs.envs import rocket_landing as RL  # noqa: E402

REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
OUTDIR = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier")


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    sty = RL.corpus_style_to_env(S["style"])
    ref = RL.PFRocketLanding.reference_env(480)
    sys.path.insert(0, os.path.join(ROOT, "scratch", "step7", "rocket-landing"))
    from measure_tier import state_quantisation_term
    sq = state_quantisation_term(ref, S, F, sty)
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"measurement": "reference_render(state13 float32) vs frozen frames (recorded from float64 physics)",
           "result": sq,
           "provenance": {"sha": sha, "baseline_commit": sha, "stable_envs_file": os.path.realpath(se.__file__),
                          "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                          "threads": 1, "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                          "renderer": "PyBullet DIRECT / TinyRenderer via PyFlyt (CPU)",
                          "versions": {k: _v(k) for k in ("PyFlyt", "pybullet", "stable-worldmodel", "numpy")}},
           "seconds": round(time.time() - t0, 1)}
    path = os.path.join(OUTDIR, f"corpus_replay_state_quantisation_{platform.node().split('.')[0]}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def _v(k):
    from importlib.metadata import version
    try:
        return version(k)
    except Exception:
        return None


if __name__ == "__main__":
    main()
