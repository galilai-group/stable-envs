"""Smoke the MiniHack env through the same checks the conformance suite runs, plus a throughput pass."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se
from stable_envs.testing import conformance as C


def main():
    print("import errors:", se.envs.import_errors())
    print("registry:", sorted(se.REGISTRY))
    env = se.make("MiniHack", batch=16, device="cuda")
    d = env.describe()
    print("describe:", {k: (v if k != "state_names" else f"<{len(v)} names>") for k, v in d.items()})
    assert len(d["state_names"]) == d["state_dim"]

    st = env.sample_states(16)
    f = env.render(st)
    print("frame", tuple(f.shape), f.dtype, "state", st.shape, st.dtype)

    print("roundtrip     ", C.check_state_roundtrip(env))
    print("batch invar   ", C.check_batch_invariance(env))
    print("determinism   ", C.check_determinism(env))
    print("variations    ", C.check_variations_do_something(env))
    t0 = time.perf_counter()
    r = C.check_state_dims_matter(env)
    print(f"dead dims     {r['dead_dims'][:8]} (n={len(r['dead_dims'])}) in {time.perf_counter()-t0:.1f}s")

    # leading dims and the preallocated out path
    s3 = torch.as_tensor(env.sample_states(6), device="cuda").view(2, 3, env.state_dim)
    f3 = env.render(s3)
    print("leading dims  ", tuple(s3.shape), "->", tuple(f3.shape))
    flat = s3.reshape(6, -1)
    buf = torch.empty(6, 21 * 16, 79 * 16, 3, dtype=torch.uint8, device="cuda")
    got = env.render(flat, out=buf)
    assert got.data_ptr() == buf.data_ptr(), "out= must write into the caller's buffer"
    assert torch.equal(got, env.render(flat)), "out= path differs from the allocating path"
    print("out= path      identical, no realloc")

    # throughput
    frame_bytes = 21 * 16 * 79 * 16 * 3
    for B in (256, 1024, 4096):
        s = torch.as_tensor(env.sample_states(64), device="cuda").repeat(B // 64, 1)
        env.render(s); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            env.render(s)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"  batch {B:5d}: {dt*1e3:7.2f} ms -> {B/dt:11.0f} frames/s "
              f"({B*frame_bytes/dt/1e9:6.1f} GB/s)")


if __name__ == "__main__":
    main()
