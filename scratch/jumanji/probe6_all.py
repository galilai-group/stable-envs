"""Probe 6: run the per-pixel-LUT construction and its verification over every candidate spec.

One line per id: conflicts, owned fraction, LUT size, and the number that decides it -- the max pixel
difference between the composed frame and the official one over random grids with EVERY cell different.
An id that reports anything but 0 there does not get a renderer; it gets a paragraph in
docs/research/jumanji.md saying which structure broke it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe5_lut import build, compose  # noqa: E402
import specs  # noqa: E402


def run(spec, size: int, n_verify: int) -> dict:
    v = spec.viewer(size)
    rf = lambda g: spec.render(v, g)
    f0 = rf(spec.base())
    if f0.shape[:2] != (size, size):
        return {"id": spec.id, "err": f"frame is {f0.shape}, asked for {size}"}
    plate, packed, stats = build(rf, spec.P, spec.n_values, spec.base(), verbose=False)

    rng = np.random.default_rng(0)
    worst_u = worst_r = 0
    for _ in range(n_verify):
        g = rng.integers(0, spec.n_values, spec.P)
        worst_u = max(worst_u, int(np.abs(compose(plate, packed, g).astype(np.int16)
                                          - rf(g).astype(np.int16)).max()))
    for _ in range(n_verify):
        g = np.asarray(spec.sampler(rng), np.int32)
        worst_r = max(worst_r, int(np.abs(compose(plate, packed, g).astype(np.int16)
                                          - rf(g).astype(np.int16)).max()))
    # how many DISTINCT colours does one symbol paint? 1 means a flat-colour palette knob is exact.
    ys, xs, cells, lut = packed
    per_symbol = [int(len(np.unique(lut[:, v], axis=0))) for v in range(spec.n_values)]
    return {"id": spec.id, **stats, "verify_uniform_max": worst_u, "verify_real_max": worst_r,
            "colors_per_symbol": per_symbol}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--verify", type=int, default=12)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    todo = [s for s in specs.ALL if args.ids is None or s.id in args.ids]
    for spec in todo:
        t0 = time.perf_counter()
        try:
            r = run(spec, args.size, args.verify)
        except Exception as e:  # noqa: BLE001
            r = {"id": spec.id, "err": f"{type(e).__name__}: {str(e)[:180]}"}
            traceback.print_exc()
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        print(r, flush=True)


if __name__ == "__main__":
    main()
