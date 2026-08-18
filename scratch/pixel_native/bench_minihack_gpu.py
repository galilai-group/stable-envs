"""The MiniHack renderer as a GPU gather: validate against the official frames, then measure it.

``frame = atlas[glyph2tile[glyphs]]`` reshaped from (H, W, 16, 16, 3) to (16H, 16W, 3).  There is no
rasterisation, so the only thing to validate is the indexing and the transpose -- against the ``pixel``
observation the env itself produced, exported by ``export_minihack_atlas.py``.

Run under SLURM with --gres=gpu:1.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def render(glyphs, atlas, g2t):
    """glyphs (B, H, W) int -> (B, 16H, 16W, 3) uint8."""
    B, H, W = glyphs.shape
    tiles = atlas[g2t[glyphs]]                       # (B, H, W, 16, 16, 3)
    t = tiles.shape[-2]
    return tiles.permute(0, 1, 3, 2, 4, 5).reshape(B, H * t, W * t, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_atlas.npz")
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 64, 256, 1024, 4096])
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_gpu.json")
    args = ap.parse_args()

    d = np.load(args.npz)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    atlas = torch.as_tensor(d["atlas"], device=dev)
    g2t = torch.as_tensor(d["glyph2tile"], device=dev, dtype=torch.long)
    glyphs = torch.as_tensor(d["glyphs"], device=dev, dtype=torch.long)
    ref = d["frames"]

    out = render(glyphs, atlas, g2t).cpu().numpy()
    exact = bool(np.array_equal(out, ref))
    worst = int(np.abs(out.astype(np.int16) - ref.astype(np.int16)).max())
    print(f"device {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")
    print(f"validated on {len(ref)} official frames {ref.shape}: bit-exact={exact} worst |diff|={worst}")

    frame_bytes = int(np.prod(ref.shape[1:]))
    timings = {}
    for B in args.batches:
        g = glyphs[torch.randint(0, glyphs.shape[0], (B,), device=dev)]
        for _ in range(3):
            r = render(g, atlas, g2t)
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            r = render(g, atlas, g2t)
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = (time.perf_counter() - t0) / n
        timings[B] = {"ms": dt * 1000, "frames_per_s": B / dt,
                      "GB_per_s": B * frame_bytes / dt / 1e9}
        print(f"  batch {B:6d}: {dt*1000:8.2f} ms -> {B/dt:12.0f} frames/s  "
              f"({B*frame_bytes/dt/1e9:7.1f} GB/s written)")
        del r

    with open(args.out, "w") as f:
        json.dump({"device": dev, "bit_exact": exact, "worst_abs_diff": worst,
                   "n_validation_frames": int(len(ref)), "frame_shape": list(ref.shape[1:]),
                   "timings": {str(k): v for k, v in timings.items()}}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
