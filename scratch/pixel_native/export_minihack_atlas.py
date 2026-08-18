"""Dump the MiniHack tile atlas, the glyph->tile table, and a batch of real glyph grids.

Two environments are involved: nle/minihack live in one venv, torch in another, so the GPU renderer
benchmark cannot import minihack.  Everything the renderer needs is three arrays, so they are exported
once here and consumed by ``bench_minihack_gpu.py``.  The reference frames are exported too, so the GPU
path is checked against the official observation rather than against itself.

Run under SLURM with the nle venv.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-5x5-v0")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_atlas.npz")
    args = ap.parse_args()

    import gymnasium as gym
    import minihack  # noqa: F401
    from minihack.tiles import glyph2tile
    from minihack.tiles.glyph_mapper import GlyphMapper

    gm = GlyphMapper()
    raw = gm.tiles
    items = list(raw.values()) if isinstance(raw, dict) else list(raw)
    atlas = np.stack([np.asarray(t, dtype=np.uint8) for t in items])
    g2t = np.asarray(glyph2tile, dtype=np.int32)

    env = gym.make(args.env, observation_keys=("glyphs", "pixel"))
    obs, _ = env.reset(seed=0)
    rng = np.random.RandomState(0)
    glyphs, frames = [], []
    for _ in range(args.n):
        glyphs.append(np.asarray(obs["glyphs"], dtype=np.int32))
        frames.append(np.asarray(obs["pixel"], dtype=np.uint8))
        obs, _r, term, trunc, _i = env.step(rng.randint(env.action_space.n))
        if term or trunc:
            obs, _ = env.reset()

    np.savez_compressed(args.out, atlas=atlas, glyph2tile=g2t,
                        glyphs=np.stack(glyphs), frames=np.stack(frames))
    print(f"atlas {atlas.shape} {atlas.dtype}, glyph2tile {g2t.shape}, "
          f"glyphs {np.stack(glyphs).shape}, frames {np.stack(frames).shape}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
