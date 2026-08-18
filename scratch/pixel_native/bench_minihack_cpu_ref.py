"""Throughput of the official MiniHack glyph->pixel path, for a like-for-like speedup number."""
from __future__ import annotations

import json
import time

import numpy as np


def main():
    from minihack.tiles.glyph_mapper import GlyphMapper

    d = np.load("/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_atlas.npz")
    glyphs = d["glyphs"]
    gm = GlyphMapper()
    gm.to_rgb(glyphs[0])                    # warm
    n = 16
    t0 = time.perf_counter()
    for i in range(n):
        gm.to_rgb(glyphs[i % len(glyphs)])
    dt = (time.perf_counter() - t0) / n
    out = {"ms_per_frame": dt * 1000, "frames_per_s": 1 / dt}
    print(f"official GlyphMapper.to_rgb: {dt*1000:.2f} ms/frame -> {1/dt:.1f} frames/s")
    p = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_cpu_ref.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
