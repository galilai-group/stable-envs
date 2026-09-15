"""Throughput per lighting band, and on the mixed corpus that is the certified number.

One number for this env would be dishonest in both directions. The daylight path is the Step-6
gather untouched -- no float buffer, no PRNG -- and keeps its throughput exactly. A night frame
carries a per-pixel float post-process that the reference performs and that this renderer reproduces
bit for bit, and no amount of engineering makes a per-pixel float chain as cheap as a memcpy. So the
bands are measured separately and the mixed corpus is reported as the certified figure, with the
band numbers beside it saying where it comes from.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from stable_envs.envs.craftax import CraftaxClassic, N_STRIP, OBS_H, OBS_W
from stable_envs.render import prng
from stable_envs.envs.craftax import night_and_sleep

OUT = "examples/results/step7/craftax-classic-full/sweep"
SLEEP = OBS_H * OBS_W * 2 + 1 + N_STRIP
LIGHT = SLEEP + 1


def bench(fn, device, n=5):
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    env = CraftaxClassic(device=a.device)
    res = {"device": a.device, "threads": torch.get_num_threads(), "torch": torch.__version__,
           "host": os.uname().nodename, "machine": os.uname().machine}
    if a.device == "cuda" and torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    batches = (1, 16, 64, 256, 1024) + ((4096,) if a.device == "cuda" else ())
    fps = {}
    for B in batches:
        for name, light, sleep in (("daylight", 1.0, 0.0), ("dusk", 0.75, None),
                                   ("night", 0.10, None), ("mixed", None, None)):
            st = env.sample_states(B, np.random.default_rng(1))
            if light is not None:
                st[:, LIGHT] = light
            if sleep is not None:
                st[:, SLEEP] = sleep
            for t in (7, 16):
                el = bench(lambda: env.render(st, size=t), a.device)
                fps[f"{name}_t{t}_B{B}"] = round(B / el)
    res["fps"] = fps

    # attribution at the certified batch, so the cost is explained and not just reported
    B, t = (256, 16) if a.device == "cpu" else (1024, 16)
    h, w = OBS_H * t, OBS_W * t
    st = env.sample_states(B, np.random.default_rng(1))
    st[:, LIGHT] = 0.10
    world = torch.rand(B, h, w, 3, device=a.device) * 255
    keys = torch.randint(-2 ** 31, 2 ** 31 - 1, (B, 2), dtype=torch.int32, device=a.device)
    light = torch.full((B,), 0.1, device=a.device)
    slp = torch.zeros(B, dtype=torch.bool, device=a.device)
    res["attribution_night"] = {
        "batch": B, "tile_px": t,
        "full_render_ms": round(bench(lambda: env.render(st, size=t), a.device) * 1e3, 2),
        "threefry_field_ms": round(bench(lambda: prng.uniform(keys, (h, w)), a.device) * 1e3, 2),
        "night_chain_ms": round(
            bench(lambda: night_and_sleep(world.clone(), light, slp, keys), a.device) * 1e3, 2),
        "world_clone_ms": round(bench(lambda: world.clone(), a.device) * 1e3, 2)}

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"bands_{a.device}{('_' + a.tag) if a.tag else ''}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
