"""Where the cuda time goes in the swm_nav family, per stage, before anything is written.

SWMNAV-2 proposes one Triton kernel for TwoRoom / TwoRoomTarget / Piecewise / PiecewiseZones. This is
the measurement that has to come first: which stage costs what on the device, how many launches a
render issues, and how much of the traffic is full-frame float32 temporaries that never reach the
output.

The stages are the env's own, timed with CUDA events and an explicit synchronise, at the batches the
frozen sweep certifies on. ``launches`` is the CUDA kernel count from the profiler, not an estimate.

    python scratch/step6/raster2d/swmnav_gpu_profile.py --device cuda --batch 256 1024 4096
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import stable_envs as se                                        # noqa: E402
from stable_envs.envs import swm_nav as SN                      # noqa: E402

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


def ev(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def launches(fn) -> tuple[int, int]:
    """``(cuda kernel launches, distinct kernels)`` for one call."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    evs = [e for e in p.key_averages() if (getattr(e, "self_device_time_total", 0) or 0) > 0]
    return sum(e.count for e in evs), len(evs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    note = {"device": a.device, "stable_envs_file": se.__file__, "host": socket.gethostname(),
            "cpu": platform.machine(), "torch": torch.__version__,
            "card": torch.cuda.get_device_name(0), "rows": []}
    print(f"tree {se.__file__}\ncard {note['card']}  host {note['host']}")
    print(f"{'config':16s}{'B':>6}{'render':>9}{'base':>8}{'dot1':>8}{'blend1':>8}{'dot2':>8}"
          f"{'blend2':>8}{'launch':>8}{'kinds':>7}   MB float temporaries")
    for cfg in CONFIGS:
        pool = np.load(ROOT / f"examples/results/step4/states/{cfg}.npz")["states"]
        for B in a.batch:
            st = np.asarray(pool[np.arange(B) % len(pool)], np.float32)
            e = se.make(cfg, batch=B, device=a.device)
            s = torch.as_tensor(st, device=e.device).reshape(-1, e.state_dim)
            k = min(B, SN.RENDER_CHUNK)                       # render() chunks at RENDER_CHUNK
            sty = e.resolve_style(None, None, batch=B)
            sl = lambda p: sty[p][0:k]                        # noqa: E731
            two = e.state_dim >= 4

            tot = ev(lambda: e.render(st), a.reps, a.warmup)
            base = ev(lambda: e._base_image(sty, 0, k), a.reps, a.warmup)
            img = e._base_image(sty, 0, k)
            d1 = ev(lambda: SN._gaussian_dot(e._gy, e._gx, s[:k, 0:2], sl("agent.radius")),
                    a.reps, a.warmup)
            dot = SN._gaussian_dot(e._gy, e._gx, s[:k, 0:2], sl("agent.radius"))
            b1 = ev(lambda: SN._alpha_blend(img, dot, sl("agent.color")), a.reps, a.warmup)
            d2 = b2 = 0.0
            if two:
                d2 = ev(lambda: SN._gaussian_dot(e._gy, e._gx, s[:k, 2:4], sl("target.radius")),
                        a.reps, a.warmup)
                b2 = ev(lambda: SN._alpha_blend(img, dot, sl("target.color")), a.reps, a.warmup)
            n, kinds = launches(lambda: e.render(st))
            # every full-frame float32 intermediate the torch path materialises, per render
            ndots = 2 if two else 1
            fl = B * SN.IMG * SN.IMG
            mb = (ndots * (fl * 4 * 3          # dx, dy, dist2 float32
                           + fl * 8 * 2        # the float64 exp argument and its result
                           + fl * 4 * 2)       # dot, dot/m
                  + ndots * (fl * 3 * 4 * 3)   # alpha_blend: img->f32, two products, the sum
                  ) / 2 ** 20
            print(f"{cfg:16s}{B:>6}{tot:9.2f}{base:8.2f}{d1:8.2f}{b1:8.2f}{d2:8.2f}{b2:8.2f}"
                  f"{n:8d}{kinds:7d}{mb:10.0f}", flush=True)
            note["rows"].append({"config": cfg, "batch": B, "render_ms": tot, "base_ms": base,
                                 "dot1_ms": d1, "blend1_ms": b1, "dot2_ms": d2, "blend2_ms": b2,
                                 "launches": n, "kernel_kinds": kinds, "float_temp_mib": mb,
                                 "chunk": k})
            del e
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(note, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
