"""The CPU-tier measurement for one cell, all three columns in ONE process on ONE node.

``examples/bench_vs_decode.py`` does this for the GPU column; the CPU tier needs the same three
numbers with OURS taken on the CPU at one thread:

    S_cpu     = stable-envs CPU f/s (1 thread)  /  official renderer f/s
    cores_cpu = stable-envs CPU f/s (1 thread)  /  one-core JPEG-q90 decode f/s of its own frames

Measuring them in separate jobs, or on separate nodes, is the different-machines violation the
certification note forbids, so all three run here back to back.

``--env`` may be a cell that does not exist in the tree being measured (the baseline leg re-measures a
FLOOR cell, not the new one); a missing cell is reported, not crashed on.

The decode timing is ``bench_vs_decode.decode_baseline``'s method reproduced exactly -- encode once at
quality 90, 20 warm-up decodes, then ``n`` timed decodes with ``n`` scaled inversely with pixel count --
so the two devices' ``cores`` columns are the same statistic.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def decode_fps_per_core(frame, quality=90, min_frames=200) -> dict:
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
    dt = (time.perf_counter() - t0) / n
    return {"h": h, "w": w, "jpeg_quality": quality, "jpeg_bytes": len(blob),
            "raw_bytes": h * w * 3, "n": n, "decode_fps_per_core": 1.0 / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--size", type=int, default=0, help="0 = the cell's registered default")
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--official-frames", type=int, default=32)
    ap.add_argument("--leg", required=True, help="baseA | branch | baseB")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    import stable_envs as se
    from stable_envs.testing import conformance as C

    out = {"leg": a.leg, "env": a.env, "host": platform.node(),
           "node_cores": int(os.sysconf("SC_NPROCESSORS_ONLN")),
           "cpus_allocated": int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or None,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "torch_threads": torch.get_num_threads(),
           "omp": os.environ.get("OMP_NUM_THREADS"),
           "loadavg_before": list(os.getloadavg()),
           "tree": se.__file__}
    import mujoco
    from importlib.metadata import version as _v
    out["mujoco"], out["dm_control"] = mujoco.__version__, _v("dm_control")

    if a.env not in se.REGISTRY:
        out["error"] = f"{a.env} is not registered in this tree"
        out["loadavg_after"] = list(os.getloadavg())
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        print(json.dumps(out, indent=1))
        return

    cls = se.REGISTRY[a.env]
    size = a.size or int(cls.default_size)
    out["size"] = size
    cells = []
    for B in [int(x) for x in a.batches.split(",")]:
        e = cls(batch=B, size=size, device="cpu", seed=0)
        st = e.sample_states(B, np.random.default_rng(31))
        t = torch.as_tensor(st)
        for _ in range(a.warmup):
            e.render(t)
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            e.render(t)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        cells.append({"batch": B, "fps": B / med, "render_ms": med * 1e3})
        print(f"[{a.leg}/{a.env}] B={B:<5} {B / med:9.2f} f/s", flush=True)
    out["cells"] = cells
    best = max(cells, key=lambda c: c["fps"])
    out["certified_batch"], out["fps"] = best["batch"], best["fps"]

    # the official renderer, one frame at a time, on THIS node
    try:
        e = cls(batch=1, size=size, device="cpu", seed=0)
        states = e.sample_states(a.official_frames, np.random.default_rng(32))
        ref = C.reference_env(cls, size)
        cls.reference_render(ref, states[:2])                       # warm the context
        t0 = time.perf_counter()
        frames = cls.reference_render(ref, states)
        dt = (time.perf_counter() - t0) / len(states)
        out["official"] = {"fps": 1.0 / dt, "frames": int(len(states))}
        print(f"[{a.leg}/{a.env}] official {1.0 / dt:.2f} f/s", flush=True)
        out["decode"] = decode_fps_per_core(frames[0])
        print(f"[{a.leg}/{a.env}] decode   "
              f"{out['decode']['decode_fps_per_core']:.1f} f/s/core", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        out["official_error"] = f"{type(exc).__name__}: {exc}"

    if out.get("official", {}).get("fps"):
        out["S_cpu"] = out["fps"] / out["official"]["fps"]
    if out.get("decode", {}).get("decode_fps_per_core"):
        out["cores_cpu"] = out["fps"] / out["decode"]["decode_fps_per_core"]
    out["loadavg_after"] = list(os.getloadavg())

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
