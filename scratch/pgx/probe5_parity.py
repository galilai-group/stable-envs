"""Parity of the GPU renderer against resvg's raster of PGX's own document.

Reports interior-pixel and all-pixel agreement over states drawn from real play, plus a breakdown of
WHERE the residual is, so a number below the bar names a primitive instead of a game.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se
from stable_envs.testing import conformance as C

GAMES = ["PgxTicTacToe", "PgxConnectFour", "PgxOthello", "PgxGo9x9", "PgxGo19x19"]


def where_is_the_residual(ours, official, env):
    """Split the disagreeing pixels into 'on a mark rim' and 'everywhere else'.

    The plate is baked through the reference, so any disagreement away from a mark is a compositing
    or geometry bug rather than an anti-aliasing difference, and that distinction is the whole point
    of looking.
    """
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    bad = d > 0
    if not bad.any():
        return {"bad": 0}
    # a pixel is "on a rim" if the reference is non-flat in its 3x3 neighbourhood
    try:
        m = C.interior_mask(official)
        flat = np.zeros_like(bad)
        flat[:, :-1, :-1] = m
    except ImportError:
        return {"bad": int(bad.sum())}
    return {"bad": int(bad.sum()),
            "bad_on_flat": int((bad & flat).sum()),
            "bad_on_rim": int((bad & ~flat).sum()),
            "worst_flat_diff": int(d[bad & flat].max()) if (bad & flat).any() else 0}


def run(names, n_states: int, sizes, supersamples) -> None:
    for nm in names:
        cls = se.REGISTRY[nm]
        for size in sizes:
            for ss in supersamples:
                env = se.make(nm, batch=8, size=size, supersample=ss, device="cuda")
                st = env.sample_states(n_states, rng=np.random.default_rng(0))
                ref = cls.reference_env(size=size)
                official = cls.reference_render(ref, st)
                ours = env.render(st).cpu().numpy()
                assert ours.shape == official.shape, f"{nm}: {ours.shape} vs {official.shape}"
                r = C.pixel_agreement(ours, official)
                w = where_is_the_residual(ours, official, env)
                ie = r["interior_exact_frac"]
                print(f"{nm:16s} size={size:4d} ss={ss} "
                      f"interior={ie if ie is None else round(ie, 6)} "
                      f"all={r['exact_frac']:.6f} mean={r['mean_abs']:.4f} "
                      f"max={r['max_abs']:3d} within8={r['within8_frac']:.5f} | {w}")
                sys.stdout.flush()
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=GAMES)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--sizes", nargs="*", type=int, default=[0])
    ap.add_argument("--ss", nargs="*", type=int, default=[1])
    a = ap.parse_args()
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"import errors: {se.envs.import_errors()}")
    print(f"registry ({len(se.REGISTRY)}): {sorted(se.REGISTRY)}\n")
    for nm in a.games:
        if nm not in se.REGISTRY:
            print(f"!! {nm} NOT REGISTERED")
    sizes = a.sizes if a.sizes != [0] else None
    for nm in a.games:
        if nm not in se.REGISTRY:
            continue
        sz = sizes or [se.REGISTRY[nm].default_size]
        run([nm], a.n, sz, a.ss)
