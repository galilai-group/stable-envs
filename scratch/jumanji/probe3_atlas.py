"""Probe 3: is the official frame a PER-PIXEL LOOKUP indexed by one cell's value?

The MiniGrid lesson generalised. There the atlas was built by CALLING ``Grid.render_tile``, so parity
was structural rather than a rasterizer-matching exercise. The same move works for any renderer whose
frame decomposes into independently-varying regions, and it does NOT require the regions to be
rectangles, to be pixel-aligned, or to be free of anti-aliasing -- which matters here, because the
reference is matplotlib and none of those three hold.

The construction, for a state of ``P`` cells each taking one of ``V`` values:

  1. render the official frame for every ``(cell p, value v)`` against a fixed base state: ``P*V`` calls;
  2. ``owner[y,x]`` = the cell whose value changes this pixel. A pixel changed by TWO cells breaks the
     model and is counted rather than tolerated;
  3. ``LUT[k, v]`` = the colour of owned pixel ``k`` when its owning cell holds ``v``;
  4. everything unowned is the static plate.

Rendering is then ``plate`` with one gather scattered into it, which is exact BY CONSTRUCTION: every
colour in the frame came out of the official renderer.

Step 5, which is the whole point, is VERIFICATION on states with many cells changed at once: the
construction above only proves each cell is separable ON ITS OWN, and a renderer that draws a shared
border, a colour bar, or a title that depends on several cells would pass steps 1-4 and fail here.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def build_lut(render_fn, P: int, V: int, base: np.ndarray, verbose: bool = True):
    """``render_fn(values) -> (H,W,3) uint8`` for a length-P integer vector.

    Returns ``(plate, owner, lut, stats)``.
    """
    t0 = time.perf_counter()
    f_base = render_fn(base)
    H, W, _ = f_base.shape
    frames = np.empty((P, V, H, W, 3), np.uint8)
    for p in range(P):
        for v in range(V):
            vals = base.copy()
            vals[p] = v
            frames[p, v] = render_fn(vals)
        if verbose:
            print(f"   cell {p + 1}/{P} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # which pixels does each cell control?
    changed = np.zeros((P, H, W), bool)
    for p in range(P):
        d = frames[p].astype(np.int16) - frames[p, base[p]].astype(np.int16)
        changed[p] = (d != 0).any(axis=(0, 3))

    n_owners = changed.sum(0)
    conflicts = int((n_owners > 1).sum())
    owner = np.where(n_owners == 1, changed.argmax(0), -1).astype(np.int32)
    owned = owner >= 0
    K = int(owned.sum())

    plate = f_base.copy()
    ys, xs = np.nonzero(owned)
    cells = owner[ys, xs]
    lut = np.empty((K, V, 3), np.uint8)
    for k in range(K):
        lut[k] = frames[cells[k], :, ys[k], xs[k]]

    stats = {"H": H, "W": W, "P": P, "V": V, "owned_px": K,
             "owned_frac": K / (H * W), "conflict_px": conflicts,
             "build_s": time.perf_counter() - t0,
             "lut_bytes": int(lut.nbytes)}
    return plate, owner, (ys, xs, cells, lut), stats


def compose(plate, packed, values: np.ndarray) -> np.ndarray:
    ys, xs, cells, lut = packed
    out = plate.copy()
    out[ys, xs] = lut[np.arange(len(ys)), values[cells]]
    return out


def verify(render_fn, plate, packed, P: int, V: int, n: int = 24, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    worst, bad = 0, 0
    for _ in range(n):
        vals = rng.integers(0, V, P)
        got = compose(plate, packed, vals)
        want = render_fn(vals)
        d = int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max())
        worst = max(worst, d)
        bad += int(d != 0)
    return {"states": n, "max_abs_diff": worst, "frames_differing": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    import adapters
    ad = adapters.get(args.id)
    plate, owner, packed, stats = build_lut(ad.render_values, ad.P, ad.V, ad.base)
    print(stats)
    print(verify(ad.render_values, plate, packed, ad.P, ad.V))


if __name__ == "__main__":
    main()
