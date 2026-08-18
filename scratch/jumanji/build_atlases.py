"""Build (and time) every Jumanji atlas at the sizes the conformance suite touches.

The suite renders at ``env.size`` and, in ``check_resolution_consistency``, at ``2 * env.size`` -- so a
cold machine pays TWO atlas builds per id, and each is ``P * V`` official matplotlib renders. This
script measures that cost per id per size, which is the number that decides whether an id can ship: the
safety plugin applies a 900 s per-test wall limit, and a build happens inside whichever test touches
the env first.

    python scratch/jumanji/build_atlases.py --sizes 200 400
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[200, 400])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from stable_envs.envs.jumanji import SPECS, _atlas_path, _build_atlas
    import numpy as np

    # Deduplicated by the atlas key, not by the id: variant ids that differ only in their reset
    # distribution (RubiksCube-partly-scrambled, Sudoku-very-easy) share the parent's atlas, and
    # building one per id would build the identical thing twice.
    uniq = {s.key_id: s for s in SPECS}.values()
    for size in args.sizes:
        for spec in uniq:
            if size not in spec.verified_sizes:
                print(f"{spec.key_id:22s} size={size:4d}  SKIPPED — the per-cell model does not hold "
                      f"at this size (verified: {spec.verified_sizes})", flush=True)
                continue
            path = _atlas_path(spec.key_id, size)
            if os.path.exists(path) and not args.force:
                print(f"{spec.key_id:22s} size={size:4d}  cached "
                      f"({os.path.getsize(path) / 1e6:.2f} MB)", flush=True)
                continue
            t0 = time.perf_counter()
            data = _build_atlas(spec, size)
            dt = time.perf_counter() - t0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **data)
            os.replace(tmp, path)
            print(f"{spec.key_id:22s} size={size:4d}  built in {dt:7.1f}s  "
                  f"renders={spec.P * spec.n_values:5d}  K={int(data['meta'][3]):7d}  "
                  f"disk={os.path.getsize(path) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
