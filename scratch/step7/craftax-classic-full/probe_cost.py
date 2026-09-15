"""Phase 0 (cost) — what the components add to a render, on this Mac (NOT a certification number).

Local Apple arm64 CPU, one torch thread. Certification numbers come from Oscar and are pending
there. The point is only to know the ORDER of the cost before proposing an implementation: the
Step-6 patch took this env from 1,867 to 115,166 f/s at B=256 on an Oscar batch node, and a night
chain that costs more than the gather would undo that.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threefry_torch as tf3               # noqa: E402
from probe_resolution import night_chain   # noqa: E402

from stable_envs.envs.craftax import CraftaxClassic, OBS_H, OBS_W   # noqa: E402

OUT = "examples/results/step7/craftax-classic-full/phase0"


def bench(fn, n=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    env = CraftaxClassic(device="cpu")
    rng = np.random.default_rng(1)
    res = {"host": os.uname().machine, "torch": torch.__version__, "threads": 1}
    for t in (7, 16):
        for B in (16, 256):
            st = env.sample_states(B, rng)
            keys = torch.as_tensor(
                rng.integers(0, 2 ** 32, size=(B, 2), dtype=np.uint64).astype(np.int64))
            H, W = OBS_H * t, OBS_W * t
            base = bench(lambda: env.render(st, size=t))
            frame = env.render(st, size=t).to(torch.float32)
            lit_day = np.ones(B, np.float32)
            lit_night = np.full(B, 0.2, np.float32)
            noise_only = bench(lambda: tf3.uniform_batch(keys, (H, W)))
            night_d = bench(lambda: night_chain(frame, lit_day, keys))
            night_n = bench(lambda: night_chain(frame, lit_night, keys))
            res[f"t{t}_B{B}"] = dict(
                world_render_ms=round(base * 1e3, 3),
                world_fps=round(B / base),
                threefry_field_ms=round(noise_only * 1e3, 3),
                night_chain_ms_day=round(night_d * 1e3, 3),
                night_chain_ms_night=round(night_n * 1e3, 3),
                night_over_world=round(night_n / base, 2),
                field_elements=B * H * W)
            print(f"t={t} B={B}", json.dumps(res[f"t{t}_B{B}"]))
    with open(os.path.join(OUT, "local_cost.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
