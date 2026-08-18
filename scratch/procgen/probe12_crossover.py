"""Where does the GPU renderer actually overtake procgen's own? Measure the crossover, not the peak.

A headline ratio taken at the GPU's best batch against the reference's best thread count is not a
lie, but on its own it hides the batch below which the GPU path is SLOWER — and this renderer has a
serial-blend structure (each composite reads the canvas the previous one wrote), which is exactly the
shape that makes small batches bad. So: sweep both sides and report the crossing.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import stable_envs as se

GAMES = {"ProcgenMaze": "maze", "ProcgenHeist": "heist", "ProcgenMiner": "miner"}
BATCHES = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NUMS = [1, 8, 32, 64, 128, 256]


def ours(name, B):
    e = se.make(name, batch=B, device="cuda")
    s = torch.as_tensor(e.sample_states(B), device="cuda")
    for _ in range(3):
        e.render(s)
    torch.cuda.synchronize()
    reps = 20 if B <= 1024 else 5
    t0 = time.time()
    for _ in range(reps):
        e.render(s)
    torch.cuda.synchronize()
    return B * reps / (time.time() - t0)


def official(game, num, threads=8):
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(num=num, env_name=game, distribution_mode="hard", rand_seed=0,
                         num_threads=threads)
    a = np.zeros(num, np.int32) + 4
    for _ in range(20):
        env.act(a)
        env.observe()
    n = max(20, min(200, 4000 // max(num, 1)))
    t0 = time.time()
    for _ in range(n):
        env.act(a)
        env.observe()
    return num * n / (time.time() - t0)


def main():
    print("device:", torch.cuda.get_device_name(0), "| cpus:", os.cpu_count())
    out = {}
    for name, game in GAMES.items():
        ref = {n: official(game, n) for n in NUMS}
        best_ref = max(ref.values())
        mine = {B: ours(name, B) for B in BATCHES}
        cross = next((B for B in BATCHES if mine[B] >= best_ref), None)
        out[name] = {"ours": mine, "official": ref, "best_official": best_ref, "crossover": cross}
        print(f"\n{name}  (official peak {best_ref:,.0f} f/s at "
              f"num={max(ref, key=ref.get)}, 8 threads)")
        print("   official: " + "  ".join(f"n{n}={ref[n]:,.0f}" for n in NUMS))
        for B in BATCHES:
            mark = " <-- crossover" if B == cross else ""
            print(f"   B={B:5d}: {mine[B]:12,.0f} f/s   {mine[B]/best_ref:7.2f}x{mark}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "crossover.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nregistry size in THIS interpreter:", len(se.REGISTRY))
    print("import errors:", sorted(se.envs.import_errors()))


if __name__ == "__main__":
    main()
