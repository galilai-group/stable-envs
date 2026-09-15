"""The single mixed-corpus tier, its five band diagnostics, and both official denominators.

The tier is one number, taken on the distribution ``sample_states(64)`` produces -- Step 3's own
protocol (``examples/bench_step3.py``), so this env's sampler IS the certification distribution.
Beside it go the five cost bands with their ratios and the slower band named, because a headline
that hides a 100x band is a headline about a state the agent is in a fraction of the time.

``S_cpu`` is reported against **both** official denominators -- the un-jitted renderer the parity
claim is made against and the jitted one Step 3 recorded for Craftax Classic -- and **neither
certifies anything until Sami rules**, because the two differ by ~500x and they are read through the
same ``reference_render`` hook.

    PYTHONPATH=. python scratch/step7/craftax-full/bench.py --device cpu --tag mac
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "examples", "results", "step7", "craftax-full", "sweep")


def throughput(env, states, size, seconds=1.2):
    env.render(states, size=size)
    if states.device.type == "cuda" if hasattr(states, "device") else False:
        torch.cuda.synchronize()
    n = len(states)
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        env.render(states, size=size)
        reps += 1
    if torch.cuda.is_available() and env.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n * reps / dt, dt / reps / n * 1e3


def decode_bar(frame, quality=90, min_frames=40):
    from PIL import Image
    h, w = int(frame.shape[0]), int(frame.shape[1])
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    blob = buf.getvalue()
    n = max(min_frames, int(4e7 / max(h * w, 1)))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    return 1.0 / ((time.perf_counter() - t0) / n), len(blob)


def band_states(cls, CF, kind, n, rng):
    """One band, built by forcing the band's own fields on a sampled batch."""
    e = cls(batch=n, device="cpu")
    st = cls.sample_states(e, n, rng)
    J = 5 * CF.N_TILE
    st[:, J + 1] = 0.0
    if kind == "level0_day":
        st[:, J + 2], st[:, J + 3] = 0, 1.0
        st[:, 4 * CF.N_TILE:5 * CF.N_TILE] = 1.0
    elif kind == "underground_lit":
        st[:, J + 2] = rng.integers(1, CF.N_LEVEL, size=n)
        st[:, 4 * CF.N_TILE:5 * CF.N_TILE] = 1.0
    elif kind == "underground_dark":
        st[:, J + 2] = rng.integers(1, CF.N_LEVEL, size=n)
        st[:, 4 * CF.N_TILE:5 * CF.N_TILE] = 0.6
    elif kind == "level0_dusk":
        st[:, J + 2], st[:, J + 3] = 0, 0.75
        st[:, 4 * CF.N_TILE:5 * CF.N_TILE] = 1.0
    elif kind == "level0_night":
        st[:, J + 2], st[:, J + 3] = 0, 0.2
        st[:, 4 * CF.N_TILE:5 * CF.N_TILE] = 1.0
    elif kind == "sleeping":
        st[:, J + 1] = 1.0
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--batches", default="1,16,256")
    ap.add_argument("--sizes", default="10,16")
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--official", action="store_true", help="also time the official renderer (needs jax)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    from stable_envs.envs import craftax_full as CF
    from stable_envs.envs.craftax_full import CraftaxFull

    rep = {"tag": a.tag, "device": a.device, "host": platform.node(),
           "platform": platform.platform(), "machine": platform.machine(),
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "note": "one torch thread; a local laptop number is NOT a certification number"}
    if a.device == "cuda" and torch.cuda.is_available():
        rep["gpu"] = torch.cuda.get_device_name(0)

    rng = np.random.default_rng(20260915)
    sizes = [int(x) for x in a.sizes.split(",")]
    batches = [int(x) for x in a.batches.split(",")]
    bands = ("level0_day", "underground_lit", "underground_dark", "level0_dusk",
             "level0_night", "sleeping")
    rep["bands"] = {}
    rep["mixed"] = {}
    for size in sizes:
        for B in batches:
            env = CraftaxFull(batch=B, device=a.device, size=size)
            mixed = CraftaxFull.sample_states(env, B, np.random.default_rng(20260915))
            mixed_t = torch.as_tensor(mixed, device=env.device)
            fps, ms = throughput(env, mixed_t, size, a.seconds)
            rep["mixed"][f"{size}px_B{B}"] = {"fps": fps, "ms_per_frame": ms}
            for kind in bands:
                st = torch.as_tensor(band_states(CraftaxFull, CF, kind, B, rng), device=env.device)
                f, m = throughput(env, st, size, a.seconds)
                rep["bands"].setdefault(f"{size}px_B{B}", {})[kind] = {"fps": f, "ms_per_frame": m}

    # ratios and the slower band, at the largest batch and the registered size
    key = f"{sizes[0]}px_B{batches[-1]}"
    b = rep["bands"][key]
    fast = max(b, key=lambda kk: b[kk]["fps"])
    slow = min(b, key=lambda kk: b[kk]["fps"])
    rep["band_summary"] = {
        "at": key, "fastest_band": fast, "slowest_band": slow,
        "ratio_fastest_over_slowest": b[fast]["fps"] / b[slow]["fps"],
        "tier_is": "the mixed-corpus number; the bands are diagnostics",
        "flagged_slower_band": f"{slow} -- the reference performs a per-pixel float post-process "
                               f"there (light map, noise field, tint, daylight blend) against a "
                               f"single uint8 gather when the frame is fully lit and awake",
    }

    env = CraftaxFull(batch=1, device=a.device, size=sizes[0])
    frame = env.render(CraftaxFull.sample_states(env, 1, rng)).cpu().numpy()[0]
    dfps, nbytes = decode_bar(frame)
    rep["decode_bar"] = {"size": sizes[0], "fps_per_core": dfps, "jpeg_q90_bytes": int(nbytes),
                         "frame_bytes": int(frame.size)}

    if a.official:
        off = {}
        import jax
        for size in sizes:
            ref = CraftaxFull.reference_env(size=size)
            e = CraftaxFull(batch=8, device="cpu", size=size)
            st = CraftaxFull.sample_states(e, 8, np.random.default_rng(7))
            t0 = time.perf_counter()
            CraftaxFull.reference_render(ref, st[:4])
            un = 4 / (time.perf_counter() - t0)
            jr = jax.jit(ref[0])
            _o, tmpl, _ = ref
            jr(tmpl).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(200):
                jr(tmpl).block_until_ready()
            jt = 200 / (time.perf_counter() - t0)
            off[str(size)] = {"unjitted_bridge_fps": un, "jitted_renderer_fps": jt,
                              "ratio": jt / un}
        rep["official"] = off
        rep["S_cpu"] = {}
        for size in sizes:
            ours = rep["mixed"][f"{size}px_B{batches[-1]}"]["fps"]
            rep["S_cpu"][str(size)] = {
                "ours_mixed_fps": ours,
                "vs_unjitted": ours / off[str(size)]["unjitted_bridge_fps"],
                "vs_jitted": ours / off[str(size)]["jitted_renderer_fps"],
                "ruling": "NEITHER certifies anything yet. The parity claim requires the un-jitted "
                          "reference and the Step-3 denominator is read through the same "
                          "reference_render hook; the two differ by ~500x. Escalated with W1's.",
            }
        rep["cores_cpu"] = {str(sizes[0]): rep["mixed"][f"{sizes[0]}px_B{batches[-1]}"]["fps"] / dfps}

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"bench_{a.device}_{a.tag}.json")
    with open(p, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2)[:6000])
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
