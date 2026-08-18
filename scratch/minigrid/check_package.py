"""Checks the conformance suite does not make, plus throughput through the public API.

``out=`` has no test because nothing in the suite passes one, and a preallocated buffer that is
silently ignored looks identical to one that works. Throughput is measured here rather than quoted
from the standalone prototype so the number describes the shipped code path.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import stable_envs as se

print(torch.cuda.get_device_name(0))

for name in ("MiniGridDoorKey", "MiniGridPOVFetch"):
    e = se.make(name, batch=16, device="cuda")
    st = e.sample_states(16)
    ref = e.render(st)

    # preallocated out, flat and with leading dims
    buf = torch.empty_like(ref)
    got = e.render(st, out=buf)
    assert got.data_ptr() == buf.data_ptr(), f"{name}: render(out=) returned a different buffer"
    assert torch.equal(buf, ref), f"{name}: render(out=) wrote the wrong pixels"

    lead = st.reshape(4, 4, -1)
    buf2 = torch.empty(4, 4, *ref.shape[1:], dtype=torch.uint8, device="cuda")
    got2 = e.render(lead, out=buf2)
    assert got2.shape == buf2.shape and torch.equal(buf2.reshape(-1, *ref.shape[1:]), ref), \
        f"{name}: render(out=) with leading dims"

    # a non-default variation must still round-trip through the same path
    v = e.render(st, variation={"palette": np.full((6, 3), 200, np.uint8)})
    assert not torch.equal(v, ref)
    print(f"{name}: out= writes through, leading dims preserved, variation live")

# throughput, on the shipped path
for name, B in (("MiniGridPOVFetch", 4096), ("MiniGridDoorKey", 4096), ("MiniGridFourRooms", 1024)):
    e = se.make(name, batch=B, device="cuda")
    st = torch.as_tensor(e.sample_states(B), device="cuda")
    out = torch.empty(B, e.size, e.size, 3, dtype=torch.uint8, device="cuda")
    for _ in range(5):
        e.render(st, out=out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        e.render(st, out=out)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    bpf = e.size * e.size * 3
    print(f"{name:20s} B={B:5d} {e.size}x{e.size}  {B/dt/1e6:6.2f}M frames/s  "
          f"{bpf*B/dt/1e9:7.1f} GB/s   state {e.state_dim*4}B vs frame {bpf}B "
          f"= {bpf/(e.state_dim*4):.0f}x")
