"""Are the disagreeing "interior" pixels actually on a boundary? Name the mechanism or don't declare.

The rate is measured: 13 of 160 seeded draws across the four swm-nav envs land below an interior floor
of exactly 1.0, always by 1-2 pixels at max |delta| = 1. That is enough to know the test is flaky and
not enough to relax anything -- this repo's rule is that a floor below 1.0 needs a stated mechanism, and
"it is only one pixel" is not a mechanism.

The hypothesis: ``interior`` means "the reference's 3x3 neighbourhood is one flat colour", and a pixel
sitting exactly on a primitive edge can satisfy that when the reference's own anti-aliasing happens to
land the whole neighbourhood on the same value. It is then classified interior while being, in fact, a
boundary pixel, and the two rasterizers round it opposite ways.

Test: for every disagreeing pixel, measure its distance to the nearest pixel where the REFERENCE image
changes value. If the hypothesis holds these are all at distance 1 -- immediately beside an edge that
the 3x3 test could not see because it looks two pixels out, not three.
"""
from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

CASES = [("TwoRoom", 25), ("TwoRoomTarget", 34), ("Piecewise", 20), ("PiecewiseZones", 2)]

for name, seed in CASES:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    cls = se.REGISTRY[name]
    env = se.make(name, batch=64, device="cuda")
    ref_env = cls.reference_env(size=env.size)
    states = env.sample_states(64, rng=np.random.default_rng(seed))
    official = cls.reference_render(ref_env, states)
    ours = env.render(states).cpu().numpy()

    diff = (ours.astype(np.int16) - official.astype(np.int16))
    bad = np.argwhere(np.abs(diff).max(-1) > 0)          # (n, 3) -> frame, y, x
    print(f"\n{name} (seed {seed}): {len(bad)} disagreeing pixels")
    for f, y, x in bad[:6]:
        g = official[f].astype(np.int16)
        # Is there an edge within 1 px? Compare the 3x3 and the 5x5 patch for constancy.
        p3 = g[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        p5 = g[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
        flat3 = bool((p3 == p3.reshape(-1, 3)[0]).all())
        flat5 = bool((p5 == p5.reshape(-1, 3)[0]).all())
        print(f"   frame {f:2d} ({y:3d},{x:3d})  |delta|={np.abs(diff[f, y, x]).max():d}  "
              f"ours={ours[f, y, x]}  ref={official[f, y, x]}  "
              f"3x3 flat={flat3}  5x5 flat={flat5}")
    if len(bad):
        f5 = 0
        for f, y, x in bad:
            g = official[f].astype(np.int16)
            p5 = g[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
            f5 += bool((p5 == p5.reshape(-1, 3)[0]).all())
        print(f"   -> {len(bad) - f5}/{len(bad)} have an edge inside the 5x5 that the 3x3 missed")
    del env
    torch.cuda.empty_cache()
