"""Throughput of the highway classes on the fused kernel, against the official pygame renderer.

Also reports the Triton compile time, because the kernel unrolls the primitive list at compile time and
these scenes carry an order of magnitude more primitives than PushT.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import statistics
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Highway"]
BATCHES = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["256", "1024", "4096"])]
print(torch.cuda.get_device_name(0))

for name in NAMES:
    env = se.make(name, batch=1, device="cuda")
    st = torch.as_tensor(env.sample_states(1), device="cuda")
    t0 = time.perf_counter()
    env.render(st)
    torch.cuda.synchronize()
    print(f"\n--- {name}  {env.H}x{env.W}px  prims={env.scene(st).n_prims}  "
          f"state_dim={env.state_dim}  compile+first call {time.perf_counter()-t0:.2f}s ---")
    del env
    for B in BATCHES:
        try:
            env = se.make(name, batch=B, device="cuda")
            st = torch.as_tensor(env.sample_states(B), device="cuda")
            out = torch.empty((B, env.H, env.W, 3), dtype=torch.uint8, device="cuda")
            for _ in range(3):
                env.render(st, out=out)
            torch.cuda.synchronize()
            ts = []
            for _ in range(8):
                torch.cuda.synchronize(); t = time.perf_counter()
                env.render(st, out=out)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            dt = statistics.median(ts)
            gb = B * env.H * env.W * 3 / 1e9
            print(f"  batch {B:6d}  {B/dt:12,.0f} frames/s   {gb/dt:6.1f} GB/s")
            del env, st, out
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"  batch {B:6d}  OOM"); torch.cuda.empty_cache()

    cls = se.REGISTRY[name]
    e = se.make(name, batch=1, device="cuda")
    ref = cls.reference_env(size=e.size)
    states = e.sample_states(64)
    cls.reference_render(ref, states[:8])
    t0 = time.perf_counter()
    cls.reference_render(ref, states)
    dt = time.perf_counter() - t0
    print(f"  official pygame renderer: {len(states)/dt:,.0f} frames/s")
