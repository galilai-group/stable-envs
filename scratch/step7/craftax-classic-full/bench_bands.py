"""Throughput on the certification distribution, and per lighting band as a diagnostic.

**There is ONE registered/default tier for this env, and it comes from the mixed distribution.** The
bands are reported beside it so the headline cannot hide the split, not as tiers of their own.

The mixed distribution is not invented here. Step 3's benchmark protocol draws
``probe.sample_states(64)`` (``examples/bench_step3.py``) -- the env's own sampler at its own seed --
so ``CraftaxClassic.sample_states`` IS the certification distribution, and its band shares are a
declared property of this cell rather than a weighting chosen for a benchmark. They are recorded
here with the numbers.

The bands differ by ~100x on CPU for a reason that lives in the reference: a lit frame carries a
per-pixel float post-process craftax performs and this renderer reproduces bit for bit, while a
full-daylight frame is one gather from a uint8 table. No engineering makes the first as cheap as the
second, so the honest thing is to publish both and let the tier come from the real mix.

Load average is recorded because it is the difference between a measurement and a number: the first
Oscar run of this script landed on a node where even the daylight path measured 6,219 f/s against
126,845 on a laptop, which is contention and not this renderer.
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
           "host": os.uname().nodename, "machine": os.uname().machine,
           "load_average": list(os.getloadavg()), "cpu_count": os.cpu_count()}
    if a.device == "cuda" and torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    # the certification distribution, recorded rather than described: Step 3 draws
    # probe.sample_states(64), so these shares ARE the registered default's state distribution
    proto = env.sample_states(64)
    L = proto[:, LIGHT]
    res["certification_distribution"] = {
        "draw": "CraftaxClassic.sample_states(64) -- examples/bench_step3.py's own protocol",
        "n": 64,
        "night_share": float((L < 0.5).mean()), "dusk_share": float(((L >= 0.5) & (L < 1.0)).mean()),
        "daylight_share": float((L == 1.0).mean()),
        "sleeping_share": float((proto[:, SLEEP] > 0.5).mean()),
        "band_assignment": "stratified by row index (i % 4), values from craftax's own day cycle"}

    batches = (1, 16, 64, 256, 1024) + ((4096,) if a.device == "cuda" else ())
    fps = {}
    for B in batches:
        # "mixed" is the sampler untouched -- the certification distribution. The three band pools
        # are the same states with light_level pinned, so a band number differs from the mixed one
        # only by the lighting and not by a different world.
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
    # the one number, and the split it must not hide
    B = 1024 if a.device == "cuda" else 256
    res["tier_inputs"] = {
        "certified_batch_note": "the sweep's own certified batch is chosen by examples/bench_step3.py;"
                                f" these are the B={B} cells for orientation",
        "batch": B,
        "mixed_fps_t16": fps[f"mixed_t16_B{B}"], "mixed_fps_t7": fps[f"mixed_t7_B{B}"],
        "daylight_fps_t16": fps[f"daylight_t16_B{B}"], "night_fps_t16": fps[f"night_t16_B{B}"],
        "daylight_over_night_t16": round(fps[f"daylight_t16_B{B}"] / fps[f"night_t16_B{B}"], 1),
        "slower_band": "night (light_level < 0.5)",
        "slower_band_why": ("a per-pixel float post-process the reference performs -- noise field, "
                            "luminance enhance, tint, daylight blend -- against a single uint8 "
                            "gather at full daylight")}

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
