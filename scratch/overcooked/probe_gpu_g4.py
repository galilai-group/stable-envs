"""G4 upper anchor: what the repo's own tilemap gather delivers on an
Overcooked-shaped atlas + index grid. Measurement only, no env written.
"""
from __future__ import annotations

import os, sys, time

import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render import tilemap as tm

dev = "cuda"
print("torch", torch.__version__, "| triton available:", tm.HAVE_TRITON,
      "|", torch.cuda.get_device_name(0))

N_TILES = 1057            # exhaustive reachable atlas incl. 5 hat colours + cook_time<=100
CASES = [
    ("cramped_room",             5, 4),
    ("counter_circuit_o_1order", 8, 5),
    ("asymmetric_advantages",    9, 5),
    ("corridor",                14, 9),
]
for T in (15, 32, 75):
    atlas = torch.randint(0, 255, (N_TILES, T, T, 3), dtype=torch.uint8, device=dev)
    print("\n-- tile_size=%d, atlas %d tiles = %.2f MB" % (T, N_TILES, atlas.numel() / 1e6))
    for name, W, H in CASES:
        for B in (256, 4096):
            idx = torch.randint(0, N_TILES, (B, H, W), dtype=torch.int32, device=dev)
            out = torch.empty(B, H * T, W * T, 3, dtype=torch.uint8, device=dev)
            for _ in range(5):
                tm.blit(atlas, idx, T, out=out)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            R = 30
            for _ in range(R):
                tm.blit(atlas, idx, T, out=out)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / R
            fps = B / dt
            gbs = out.numel() * 2 / dt / 1e9   # read + write
            print("   %-26s %2dx%2d -> %4dx%4d  B=%5d : %10.0f f/s   %6.1f GB/s"
                  % (name, W, H, W * T, H * T, B, fps, gbs))
print("\nDONE")
