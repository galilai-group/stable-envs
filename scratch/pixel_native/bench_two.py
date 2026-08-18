"""Throughput of the two envs landed here, on one GPU, at their reference tile sizes."""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se


def bench(name, batches=(256, 1024, 4096)):
    env = se.make(name, batch=16, device="cuda")
    f = env.render(env.sample_states(4))
    fb = int(f[0].numel())
    print(f"\n{name}: frame {tuple(f.shape[1:])} = {fb} B, state {env.state_dim} floats = "
          f"{4*env.state_dim} B  ({fb/(4*env.state_dim):.0f}x)")
    for B in batches:
        s = torch.as_tensor(env.sample_states(64), device="cuda").repeat(B // 64, 1)
        env.render(s); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            env.render(s)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"  batch {B:5d}: {dt*1e3:7.2f} ms -> {B/dt:12.0f} frames/s ({B*fb/dt/1e9:6.1f} GB/s)")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    bench("MiniHack")
    bench("CraftaxClassic", batches=(1024, 4096, 16384))
