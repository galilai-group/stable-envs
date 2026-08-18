"""Localise the CraftaxClassic vs official disagreement, and check whether a tile size has interiors.

Two things go wrong at once on the first parity run and they are unrelated, so they are separated here:

  * 0.24% of pixels differ with max |diff| 255, which is a whole wrong tile somewhere, not rounding.
    Reported per tile and cross-referenced against the mob/block id at that tile, so the culprit is
    named rather than guessed at.
  * ``interior_frac`` came back 0.0, i.e. the conformance suite found no pixel whose 3x3 neighbourhood
    in the reference is flat, so its interior statistic is NaN and the parity assertion cannot pass at
    all. That is a property of 7 px pixel art, not of this renderer, so the same measurement is repeated
    at 16 px (``BLOCK_PIXEL_SIZE_IMG``, craftax's other declared tile size).
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")


def report(tile_px):
    import torch

    import stable_envs as se
    from stable_envs.envs import craftax as CX
    from stable_envs.testing.conformance import interior_mask, pixel_agreement

    env = se.make("CraftaxClassic", batch=16, device="cuda", size=tile_px)
    st = env.sample_states(24)

    ref = CX.CraftaxClassic.reference_env(size=tile_px)
    official = CX.CraftaxClassic.reference_render(ref, st)
    ours = env.render(st).cpu().numpy()
    print(f"\n== tile {tile_px}px == ours {ours.shape} official {official.shape}")
    r = pixel_agreement(ours, official)
    print("  ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    m = interior_mask(official)
    print(f"   interior pixels: {m.mean()*100:.3f}% of frame")

    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    bad = d > 0
    if bad.any():
        n, H, W = bad.shape
        tiles = bad.reshape(n, CX.OBS_H, tile_px, CX.OBS_W, tile_px).any(axis=(2, 4))
        nb = CX.OBS_H * CX.OBS_W
        idx = np.argwhere(tiles)
        print(f"   {len(idx)} disagreeing tiles over {n} frames; first 12:")
        for f, r_, c_ in idx[:12]:
            blk = int(round(float(st[f, r_ * CX.OBS_W + c_])))
            mob = int(round(float(st[f, nb + r_ * CX.OBS_W + c_])))
            print(f"     frame {f:2d} tile ({r_},{c_})  block={blk:2d} mob={mob}  "
                  f"centre={(r_, c_) == (CX.OBS_H // 2, CX.OBS_W // 2)}")
        mobs = [int(round(float(st[f, nb + r_ * CX.OBS_W + c_]))) for f, r_, c_ in idx]
        import collections
        print("   mob ids on disagreeing tiles:", collections.Counter(mobs))
    else:
        print("   no disagreeing tiles")


if __name__ == "__main__":
    for t in (7, 16):
        report(t)
