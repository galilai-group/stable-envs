"""Phase 0 — what adding the three components does to ``check_resolution_consistency``.

The frozen Step-4 record declares ``DOWNSAMPLE_EXACT_FRAC >= 1.0 - 0.005`` for ``CraftaxClassic``
and observed **1.0**: rendering the world layer at 32 px and box-downsampling to 16 px reproduces
the 16 px render exactly, because craftax's nearest resampling from 16 px art to 32 px is exact 2x
pixel replication.

Neither new component inherits that property, and this measures by how much rather than asserting:

* the **strip** sizes its icons at ``int(0.8*bps)`` and its digits at ``int(0.6*bps)``. At 16 px
  that is 12 and 9; at 32 px it is **25 and 19**, not 24 and 18. The strip at 2S is a genuinely
  different picture, not a supersample of the S one.
* the **night noise field** is drawn at ``(7t, 9t)``, so element ``i`` of the 32 px field is a
  different threefry counter from element ``i`` of the 16 px field. Downsampling averages four
  independent samples against one.

Uses the worktree's own (unmodified) ``CraftaxClassic`` for the world layer. Nothing is edited. The
night chain is applied to the uint8 world frame rather than to the pre-rounding float composite, so
these numbers are a diagnostic of the RESAMPLING mismatch, not the shipping arithmetic; the
resampling mismatch is what the check measures and it is unaffected by where the rounding sits.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threefry_torch as tf3          # noqa: E402

from stable_envs.envs.craftax import CraftaxClassic, OBS_H, OBS_W   # noqa: E402

OUT = "examples/results/step7/craftax-classic-full/phase0"
N_CELL = 16
N_COUNT = 10


def box_downsample(x, f):
    B, H, W, C = x.shape
    return (x.reshape(B, H // f, f, W // f, f, C).astype(np.float64)
            .mean(axis=(2, 4)).round().astype(np.uint8))


def vignette(H, W):
    xs, ys = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    v = 1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (0.5 ** 2)).T
    return np.repeat(v[:, :, None], 3, axis=-1).astype(np.float32)


def night_chain(world, light, keys):
    """``world`` (B,H,W,3) float32 torch; ``light`` (B,) float; ``keys`` (B,2) int64."""
    B, H, W, _ = world.shape
    night_tex = torch.tensor([0.0, 16.0, 64.0]).view(1, 1, 1, 3)
    vg = torch.as_tensor(vignette(H, W)).view(1, H, W, 3)
    lt = torch.as_tensor(light, dtype=torch.float32).view(B, 1, 1, 1)
    inten = torch.clamp_min(2.0 * (0.5 - lt), 0.0)
    noise = tf3.uniform_batch(keys, (H, W)) * 95 + 32
    mask = inten * vg
    nws = (1 - mask) * world + mask * noise[:, :, :, None]
    npx = torch.where(lt < 0.5, nws, world)
    lum = 0.299 * npx[..., 0] + 0.587 * npx[..., 1] + 0.114 * npx[..., 2]
    npx = npx * 0.4 + (1 - 0.4) * lum[..., None]
    npx = 0.5 * npx + 0.5 * night_tex
    return lt * world + (1 - lt) * npx


def main():
    os.makedirs(OUT, exist_ok=True)
    env = CraftaxClassic(device="cpu")
    rng = np.random.default_rng(20260915)
    st = env.sample_states(16, rng)
    keys = torch.as_tensor(rng.integers(0, 2 ** 32, size=(16, 2), dtype=np.uint64).astype(np.int64))
    res = {}

    for S, F in ((16, 2),):
        # world layer only, as it is today (the frozen 1.0)
        a = env.render(st, size=S).numpy()
        big = env.render(st, size=S * F).numpy()
        d = np.abs(box_downsample(big, F).astype(np.int16) - a.astype(np.int16))
        res["world_only_today"] = dict(size=S, factor=F, exact_frac=float((d.max(-1) == 0).mean()),
                                       mean_abs=float(d.mean()), max_abs=int(d.max()))

        # world layer with the night chain applied, at both sizes
        for light in (1.0, 0.75, 0.45, 0.1):
            lv = np.full(16, light, np.float32)
            fa = night_chain(torch.as_tensor(a.astype(np.float32)), lv, keys)
            fb = night_chain(torch.as_tensor(big.astype(np.float32)), lv, keys)
            ua = fa.round().clamp(0, 255).to(torch.uint8).numpy()
            ub = fb.round().clamp(0, 255).to(torch.uint8).numpy()
            d = np.abs(box_downsample(ub, F).astype(np.int16) - ua.astype(np.int16))
            res[f"world_plus_night_light_{light}"] = dict(
                exact_frac=float((d.max(-1) == 0).mean()), mean_abs=float(d.mean()),
                max_abs=int(d.max()))

    # strip geometry at a size sweep: is 2S's strip a 2x replication of S's?
    geo = {}
    for bps in (7, 8, 16, 32, 64):
        geo[bps] = dict(icon=int(0.8 * bps), number=int(0.6 * bps),
                        icon_lo=(bps - int(0.8 * bps)) // 2 - 1,
                        number_offset=bps - int(0.6 * bps))
    res["strip_geometry_sweep"] = geo
    res["strip_scales_by_exact_2x"] = {
        f"{s}->{2*s}": bool(geo[2 * s]["icon"] == 2 * geo[s]["icon"]
                            and geo[2 * s]["number"] == 2 * geo[s]["number"])
        for s in (8, 16, 32) if s in geo and 2 * s in geo}
    print(json.dumps(res, indent=2))
    with open(os.path.join(OUT, "resolution_consistency_risk.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
