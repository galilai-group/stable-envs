"""Probe 9: the 2S->S box-downsample agreement, per env.

``tests/gpu/test_reproducibility.py`` refuses to assert a bound it was not given a MEASURED value for
(``DOWNSAMPLE_EXACT_FRAC``), which is the right design and means a new env has to produce the number
before it can be recorded. This produces it, using the same helper the test uses and the same states.
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

warnings.filterwarnings("ignore")

import numpy as np


def main() -> None:
    import stable_envs as se
    from stable_envs.testing import conformance as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=None)
    args = ap.parse_args()

    names = [n for n in sorted(se.REGISTRY)
             if n.startswith("Jumanji") and (args.envs is None or n in args.envs)]
    for name in names:
        env = se.make(name, batch=8, device="cuda")
        states = env.sample_states(8, rng=np.random.default_rng(0))
        try:
            r = C.check_resolution_consistency(env, states)
        except ValueError as e:      # the env renders at one size only, and says so
            print(f'    # {name}: no 2x atlas — {str(e).split(".")[0]}')
            continue
        if "skipped" in r:
            print(f'    # {name}: skipped ({r["skipped"]})')
            continue
        print(f'    "{name}":{" " * max(1, 30 - len(name))}{r["exact_frac"]:.6f},   '
              f'# mean {r["mean_abs"]:.4f}/255', flush=True)


if __name__ == "__main__":
    main()
