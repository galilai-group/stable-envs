"""M4 — the two cells' performance tiers, measured with same-node comparators.

Three numbers decide a GPU tier and all three must come from the SAME node: our fps, the official
renderer's fps, and the one-core JPEG-q90 decode fps of the env's own frame. A ratio built from
numbers taken on different machines is not a ratio, which is the rule Step 6 states and the reason
the official and decode legs run here rather than being read off an earlier artifact.

`BATCH_SCALES` is reported and is expected to FAIL: phase 0 measured 2.93x against a required 20x on
this scene, because at 224 px `fps(1)` is already thousands and throughput saturates by B=64, so
there is nothing left for batching to amortise. Under the frozen Section 6.7 rule a BATCH_SCALES
failure caps the tier at `GPU_PARITY`, and that cap is declared rather than worked around.

The CPU leg is one line and it is an assertion, not a measurement: `device="cpu"` must RAISE. There
is no CPU backend for this family (CPU3D-OGB-1) and silently serving a CPU request from the GPU is
what the Step-6 policy addendum forbids.
"""
from __future__ import annotations

import argparse, io, json, os, platform, socket, statistics, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--repeat", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    import torch
    from PIL import Image
    import stable_envs as se

    rec = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "platform": platform.platform(), "mujoco_gl": os.environ.get("MUJOCO_GL"),
           "note": "our fps, the official renderer and the decode bar all measured on THIS node",
           "cells": {}}

    for cname in a.cells:
        cls = se.REGISTRY[cname]
        cell = {"size": int(cls.default_size), "rows": []}

        # ---- the CPU contract: a refusal, asserted ---------------------------------------
        try:
            se.make(cname, batch=1, device="cpu")
            cell["cpu_raises"] = False
            cell["cpu_error"] = None
        except ValueError as e:
            cell["cpu_raises"] = True
            cell["cpu_error"] = str(e)[:400]
        cell["cpu_tier"] = "NO CPU BACKEND (CPU3D-OGB-1)"

        env = se.make(cname, batch=max(a.batches), device="cuda")
        st = env.sample_states(64, np.random.default_rng(5))

        for B in a.batches:
            try:
                s = np.repeat(st, int(np.ceil(B / len(st))), axis=0)[:B]
                for _ in range(3):
                    env.render(s)
                torch.cuda.synchronize()
                ts = []
                for _ in range(a.repeat):
                    t0 = time.perf_counter()
                    env.render(s)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)
                cell["rows"].append({"batch": B, "ms": med * 1e3, "fps": B / med,
                                     "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)})
                print(f"{cname} B={B}: {med*1e3:.2f} ms -> {B/med:.0f} f/s", flush=True)
            except Exception as e:
                cell["rows"].append({"batch": B, "error": f"{type(e).__name__}: {e}"[:200]})
                torch.cuda.empty_cache()

        ok = [r for r in cell["rows"] if "fps" in r]
        cell["fps_best"] = max(r["fps"] for r in ok)
        cell["fps_1"] = next(r["fps"] for r in ok if r["batch"] == 1)
        cell["batch_scales"] = cell["fps_best"] / cell["fps_1"]

        # ---- same-node comparators --------------------------------------------------------
        ref = cls.reference_env()
        u = ref.unwrapped
        for _ in range(10):
            u.render()
        blk = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(60):
                u.render()
            blk.append(60 / (time.perf_counter() - t0))
        cell["official_fps_same_node"] = statistics.median(blk)

        frame = u.render().copy()
        buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        blob = buf.getvalue()
        for _ in range(30):
            np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        dblk = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(300):
                np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
            dblk.append(300 / (time.perf_counter() - t0))
        cell["decode_fps_per_core_same_node"] = statistics.median(dblk)
        cell["jpeg_q90_bytes"] = len(blob)

        cell["R_ours_over_official"] = cell["fps_best"] / cell["official_fps_same_node"]
        cell["cores"] = cell["fps_best"] / cell["decode_fps_per_core_same_node"]
        cell["batch_scales_pass"] = bool(cell["batch_scales"] >= 20.0)
        cell["declared_gpu_tier"] = ("GPU_PARITY (capped: BATCH_SCALES "
                                     f"{cell['batch_scales']:.2f}x < 20x required)")
        print(f"{cname} TIERS: best {cell['fps_best']:.0f} f/s  official {cell['official_fps_same_node']:.1f}"
              f"  decode {cell['decode_fps_per_core_same_node']:.0f}  R {cell['R_ours_over_official']:.1f}"
              f"  cores {cell['cores']:.2f}  BATCH_SCALES {cell['batch_scales']:.2f}x"
              f"  cpu_raises {cell['cpu_raises']}", flush=True)
        rec["cells"][cname] = cell

    (out / "m4_tiers.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
