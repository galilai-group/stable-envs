"""MiniHack / NetHack: the frame is an atlas blit of the glyph grid -- verified by pixel diff.

``minihack.tiles.glyph_mapper`` maps glyph id -> tile id -> a fixed RGB tile, so the claim is that the
``pixel`` observation is exactly ``tiles[glyph2tile[glyphs]]`` tiled out.  That is checked the same way
the PushT GPU renderer was checked: run the official env, take the observation it produces, rebuild it
from the compact state alone, and diff.

Also measured: the size of the glyph grid against the size of the frame, and the size of the tile atlas.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-5x5-v0")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_state.json")
    args = ap.parse_args()

    import gymnasium as gym
    import minihack  # noqa: F401
    from minihack.tiles import glyph2tile
    from minihack.tiles.glyph_mapper import GlyphMapper
    from nle import nethack

    gm = GlyphMapper()
    raw = gm.tiles
    items = list(raw.values()) if isinstance(raw, dict) else list(raw)
    tiles = np.stack([np.asarray(t) for t in items])
    print(f"atlas: container={type(raw).__name__} n={len(items)} stacked={tiles.shape} {tiles.dtype} "
          f"({tiles.nbytes/1024:.0f} kB in RAM)")
    print(f"glyph2tile: {len(glyph2tile)} entries, MAX_GLYPH={nethack.MAX_GLYPH}, "
          f"distinct tile ids used = {len(set(glyph2tile))}")

    env = gym.make(args.env, observation_keys=("glyphs", "pixel", "chars", "colors"))
    obs, _ = env.reset(seed=0)
    rng = np.random.RandomState(0)

    exact, maxdiff = [], []
    for t in range(args.n_steps):
        g = np.asarray(obs["glyphs"])
        px = np.asarray(obs["pixel"])
        rebuilt = gm.to_rgb(g)
        ok = bool(np.array_equal(rebuilt, px))
        exact.append(ok)
        maxdiff.append(int(np.abs(rebuilt.astype(np.int16) - px.astype(np.int16)).max()) if not ok else 0)
        obs, *_rest = env.step(rng.randint(env.action_space.n))
        term, trunc = _rest[1], _rest[2]
        if term or trunc:
            obs, _ = env.reset()

    g = np.asarray(obs["glyphs"])
    px = np.asarray(obs["pixel"])
    out = {
        "env": args.env,
        "glyph_grid_shape": list(g.shape),
        "glyph_dtype": str(g.dtype),
        "glyph_bytes": int(g.nbytes),
        "glyph_bytes_as_uint16": int(np.prod(g.shape) * 2),
        "frame_shape": list(px.shape),
        "frame_bytes": int(px.nbytes),
        "compression_frame_over_glyphs": float(px.nbytes) / float(np.prod(g.shape) * 2),
        "atlas_shape": list(tiles.shape),
        "atlas_n_tiles": int(tiles.shape[0]),
        "atlas_bytes": int(tiles.nbytes),
        "n_glyphs": int(nethack.MAX_GLYPH),
        "n_distinct_tiles": int(len(set(glyph2tile))),
        "tile_px": [int(tiles.shape[1]), int(tiles.shape[2])],
        "n_steps": args.n_steps,
        "frac_frames_bit_exact_from_glyphs": float(np.mean(exact)),
        "worst_abs_diff": int(max(maxdiff)),
        "distinct_glyphs_seen": None,
    }
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
