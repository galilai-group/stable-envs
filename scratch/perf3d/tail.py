"""What is left in a DM Control chunk once the ray passes are fused and the windows are bounded.

Times the composite tail on its own — plate promotion, the box filter, the round/clamp/cast — against
the traffic each of them has to move, so the next change is aimed at whichever one is furthest from
its own bandwidth bound rather than at whichever one looks biggest in a profile.
"""
from __future__ import annotations

import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--ss", type=int, default=2)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()

B, size, ss = a.batch, a.size, a.ss
R = size * ss
N = R * R
dev = "cuda"
plate = torch.randint(0, 256, (R, R, 3), device=dev, dtype=torch.uint8)
win = 65536
widx = torch.randint(0, N, (B, win), device=dev, dtype=torch.long).sort(1).values
w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
col_w = torch.rand(B, win, 3, device=dev) * 255
cov_w = torch.rand(B, win, device=dev) < 0.05


def t_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / a.reps


def mb(x):
    return x / 1e6


base_bytes = B * N * 3 * 4
print(f"{torch.cuda.get_device_name(0)}   B={B} {size}px ss={ss}  "
      f"one (B,N,3) float32 = {mb(base_bytes):.0f} MB, output {mb(B * size * size * 3):.1f} MB\n")

cases = {}
cases["plate -> float (B,N,3)"] = (
    lambda: plate.reshape(1, N, 3).expand(B, N, 3).to(torch.float32), 2 * base_bytes)
base = plate.reshape(1, N, 3).expand(B, N, 3).to(torch.float32)
cases["gather window"] = (lambda: base.gather(1, w3), 2 * win * B * 3 * 4)
patch = base.gather(1, w3)
cases["scatter_ window"] = (lambda: base.scatter_(1, w3, patch), 2 * win * B * 3 * 4)
cases["box filter sum((2,4))"] = (
    lambda: base.view(B, size, ss, size, ss, 3).sum((2, 4)), base_bytes * 1.25)
f = base.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
cases["round/clamp/uint8"] = (
    lambda: torch.clamp(torch.round(f), 0, 255).to(torch.uint8), 3 * f.numel() * 4)
cases["where(cover, col, base) full"] = (
    lambda: torch.where((base[..., 0] < 128).unsqueeze(-1), base, base), 3 * base_bytes)

tot = 0.0
for lbl, (fn, byts) in cases.items():
    dt = t_of(fn)
    tot += dt
    print(f"  {lbl:32s} {dt * 1e3:8.3f} ms   {byts / dt / 1e9:7.0f} GB/s")
print(f"  {'sum of the above':32s} {tot * 1e3:8.3f} ms")
