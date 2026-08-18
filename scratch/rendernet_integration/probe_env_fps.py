"""Throughput at the ENV's OWN surface — states in, uint8 frames out — for both decoders.

The compile probe timed the composition by hand. This times what a consumer calls, so the buffer
allocation, the chunk loop and the corrector's static-buffer copies are all inside the number, and the
batch ladder shows what the network's fixed batch costs a caller who asks for fewer frames than it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

#: 8-worker Lance read per cell, from ``examples/results/storage/lance.json``.
DISK = {"RoboCasaCounterToSink": 15937}


def timed(fn, reps: int):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return float(np.median(a)), 100 * float(np.percentile(a, 75) - np.percentile(a, 25)) / float(
        np.median(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="RoboCasaCounterToSink")
    ap.add_argument("--batches", default="1,16,64,256,1024")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/"
                                     "rendernet/integration_env_fps.json")
    a = ap.parse_args()
    import stable_envs as se
    print(se.format_registry(), flush=True)
    rows = []
    for name in (a.plain, a.plain + "Corrected"):
        env = se.make(name, batch=1, size=256, device="cuda", seed=0)
        pool = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
        for b in [int(x) for x in a.batches.split(",")]:
            idx = torch.as_tensor(np.random.default_rng(0).integers(0, 64, b), device="cuda")
            st = pool[idx]
            buf = torch.empty((b, 256, 256, 3), dtype=torch.uint8, device="cuda")
            dt, iqr = timed(lambda: env.render(st, out=buf), a.reps)
            rows.append({"env": name, "decoder": env.render_decoder, "batch": b, "fps": b / dt,
                         "iqr_pct": iqr, "disk_fps": DISK.get(a.plain)})
            print(json.dumps(rows[-1]), flush=True)
            del buf
            torch.cuda.empty_cache()
        del env
    for b in sorted({r["batch"] for r in rows}):
        p = next(r["fps"] for r in rows if r["batch"] == b and r["decoder"] == "renderer")
        c = next(r["fps"] for r in rows if r["batch"] == b and r["decoder"] == "learned")
        print(f"batch {b:5d}: renderer {p:9.1f} f/s   learned {c:9.1f} f/s   "
              f"x{c / p:.2f}   vs disk {c / DISK[a.plain]:.2f}x", flush=True)
    with open(a.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "rows": rows}, f, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
