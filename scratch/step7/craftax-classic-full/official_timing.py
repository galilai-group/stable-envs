"""Time the official Craftax renderer the way craftax runs it, on the certification host.

Sami's ruling: the reference throughput is the official renderer **executed the way craftax executes
it -- jitted, warm, compile excluded -- timed in the same job on the same host as ours**; parity
stays against the un-jitted function as a uint8 claim. So this times the jitted one and our renderer
in a single process, and records the conditions rather than assuming them.

**Why the existing numbers could not be used.** ``bench_step3`` already reports ``official_cpu``, and
on the branch it reads 2.18 f/s because :meth:`CraftaxClassic.reference_env` returns the renderer
un-jitted -- correct for parity, wrong for a ratio. The baseline leg's 610.6 / 386.7 f/s ARE jitted
and on the right host, but neither meets the ruling's conditions:

* **CPU** -- our renderer was pinned to one core (``--threads 1``, ``OMP_NUM_THREADS=1``) and jax was
  not: XLA's CPU backend runs its own thread pool and ignores those variables. A ratio of a
  one-core renderer to a many-core reference is not a ratio. Here BOTH sides run under
  ``taskset -c 0``, so "one core" is enforced by the OS rather than requested of a library.
* **GPU** -- ``env_header.sh`` exports ``JAX_PLATFORMS=cpu``, which both jobs source, so the
  reference never touched the card it was supposedly compared on. Here jax runs on the same device
  as torch, with preallocation off so the two can share it.

Emits the reference fps plus the provenance needed to judge it: jax backend and devices, the
pinning, torch threads, and our own throughput on the same states in the same process.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from stable_envs.envs.craftax import CraftaxClassic

OUT = "examples/results/step7/craftax-classic-full/sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=("cpu", "cuda"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--batch", type=int, default=0, help="our certified batch; 0 picks per device")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    B = a.batch or (64 if a.device == "cpu" else 4096)

    import jax
    env = CraftaxClassic(device=a.device)
    pool = env.sample_states(a.frames + 8)          # the certification distribution

    res = {"device": a.device, "size": a.size, "frames": a.frames, "our_batch": B,
           "host": os.uname().nodename, "machine": os.uname().machine,
           "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
           "jax": jax.__version__, "jax_default_backend": jax.default_backend(),
           "jax_devices": [str(d) for d in jax.devices()],
           "affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "env": {k: os.environ.get(k) for k in
                   ("JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE")}}
    if a.device == "cuda" and torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    # --- the official renderer, JITTED: the way craftax runs it
    fn, tmpl, size = CraftaxClassic.reference_env(size=a.size)
    jitted = (jax.jit(fn), tmpl, size)
    CraftaxClassic.reference_render(jitted, pool[:8])                 # warm: compile excluded
    t0 = time.perf_counter()
    CraftaxClassic.reference_render(jitted, pool[8:8 + a.frames])
    dt = (time.perf_counter() - t0) / a.frames
    res["official_jitted"] = {"fps": 1 / dt, "seconds_per_frame": dt}

    # --- and un-jitted, for the record: this is what reference_env returns and what parity uses
    CraftaxClassic.reference_render((fn, tmpl, size), pool[:2])
    t0 = time.perf_counter()
    CraftaxClassic.reference_render((fn, tmpl, size), pool[8:8 + 4])
    res["official_unjitted"] = {"fps": 4 / (time.perf_counter() - t0), "frames": 4}

    # --- ours, same states, same process, same pinning
    st = np.asarray(pool[:B] if len(pool) >= B else np.repeat(pool, B // len(pool) + 1, 0)[:B],
                    np.float32)
    env.render(st, size=a.size)
    if a.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        env.render(st, size=a.size)
    if a.device == "cuda":
        torch.cuda.synchronize()
    res["ours"] = {"fps": B * 3 / (time.perf_counter() - t0), "batch": B}
    res["ratio_ours_over_official_jitted"] = res["ours"]["fps"] / res["official_jitted"]["fps"]

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"official_timing_{a.device}{('_' + a.tag) if a.tag else ''}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
