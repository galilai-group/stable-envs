"""Separate the two candidate causes of the DDP mismatch: batch size, or device.

The DDP test renders 64 states as one batch in the reference and as four batches of 16 in the ranks,
AND on two different GPUs. Those are two changes at once, so the failure it reports does not say which
one moved. This does the two one-factor comparisons.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import stable_envs as se

NAMES = sys.argv[1:] or ["PushT", "MiniGridEmpty", "PointmazeMediumNavigate",
                         "PointmazeLargeNavigate", "AntmazeMediumNavigate"]


def render_in_chunks(name, states, batch, device):
    out = []
    for i in range(0, len(states), batch):
        env = se.make(name, batch=min(batch, len(states) - i), device=device)
        out.append(env.render(states[i:i + batch]).cpu())
        del env
        torch.cuda.empty_cache()
    return torch.cat(out)


def diff(a, b):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return int(d.max()), int((d > 0).any(-1).sum())


for name in NAMES:
    if name not in se.REGISTRY:
        print(f"{name:26s} not registered")
        continue
    rng = np.random.default_rng(0)
    proto = se.make(name, batch=1, device="cuda:0")
    states = proto.sample_states(64, rng=rng)
    del proto
    torch.cuda.empty_cache()

    full0 = render_in_chunks(name, states, 64, "cuda:0")
    chunk0 = render_in_chunks(name, states, 16, "cuda:0")
    full1 = render_in_chunks(name, states, 64, "cuda:1") if torch.cuda.device_count() > 1 else None

    mb, nb = diff(full0, chunk0)
    line = f"{name:26s} batch 64 vs 16 on cuda:0 -> max {mb}/255 on {nb} px"
    if full1 is not None:
        md, nd = diff(full0, full1)
        line += f"   |   cuda:0 vs cuda:1 at batch 64 -> max {md}/255 on {nd} px"
    # A repeat on the same device at the same batch is the control: if THIS is non-zero the renderer
    # is not deterministic at all and neither comparison above means anything.
    again = render_in_chunks(name, states, 64, "cuda:0")
    mr, nr = diff(full0, again)
    line += f"   |   repeat -> max {mr}/255 on {nr} px"
    print(line, flush=True)
