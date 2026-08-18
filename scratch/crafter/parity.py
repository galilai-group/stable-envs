"""Crafter GPU renderer vs the official frames harvested by harvest.py, plus throughput."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.testing import conformance as C

d = np.load("/mnt/vast/home/randall.balestriero/se_logs/crafter_pool.npz")
st, official = d["states"], d["frames"]
env = se.make("Crafter", batch=16, device="cuda")
ours = env.render(st).cpu().numpy()
print("shape", ours.shape, official.shape)
r = C.pixel_agreement(ours, official)
print("parity over the whole 512-state pool:", r)

diff = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
bad = np.flatnonzero(diff.reshape(len(st), -1).max(1))
print(f"frames with any disagreement: {len(bad)}/{len(st)}  {bad[:8]}")

# throughput
for B in (256, 1024, 4096, 16384):
    s = torch.as_tensor(env.sample_states(B), device="cuda")
    out = torch.empty(B, 64, 64, 3, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        env.render(s, out=out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        env.render(s, out=out)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"B={B:6d}  {B/dt:12,.0f} frames/s  {B*64*64*3/dt/1e9:6.1f} GB/s")

# the official CPU renderer, for the ratio: a LIVE env rendering its own observation, which is what
# a user's rollout pays, plus the reference bridge's rate (which also rebuilds the world per frame)
try:
    ref = C.reference_env(type(env), 64)
    t0 = time.perf_counter()
    type(env).reference_render(ref, st[:64])
    dt = (time.perf_counter() - t0) / 64
    print(f"official reference bridge (world rebuild + render): {1/dt:,.0f} f/s")

    import crafter
    live = crafter.Env(seed=0)
    live.reset()
    for _ in range(8):
        live.render()
    t0 = time.perf_counter()
    for _ in range(200):
        live.render()
    dt = (time.perf_counter() - t0) / 200
    print(f"official crafter.Env.render on a live env: {1/dt:,.0f} f/s")
except ImportError as e:
    print("crafter not installed here:", e)
