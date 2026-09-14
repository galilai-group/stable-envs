"""Every torch/numpy spelling of "three uint8 planes -> one interleaved (B,H,W,3), rows flipped".

This is the largest single item in the after-profile of the planar canvas and the number the
GFXDRAW-2-CPU proposal turns on, so it gets its own probe rather than a figure quoted from a
session. Run it on the host you intend to quote: the same code on the same laptop measured 10.7 ms
in one session and 18.5 in another, which is the whole reason this file exists.

    python scratch/step6/gfxdraw/interleave_bench.py --out <path>
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="64,400,600", help="B,H,W")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    B, H, W = (int(v) for v in a.shape.split(","))

    A = torch.randint(0, 255, (3, B, H, W), dtype=torch.uint8)
    out = torch.empty(B, H, W, 3, dtype=torch.uint8)
    An, outn = A.numpy(), out.numpy()
    p32 = torch.randint(0, 1 << 24, (B, H, W), dtype=torch.int32)
    v = p32.view(torch.uint8).view(B, H, W, 4)

    def np_assign():
        for c in range(3):
            outn[..., c] = An[c, :, ::-1]

    cases = {
        "torch: 3x out[...,c].copy_(plane.flip(1))  [shipped]":
            lambda: [out[..., c].copy_(A[c].flip(1)) for c in range(3)],
        "torch: stack of flipped planes":
            lambda: torch.stack((A[0].flip(1), A[1].flip(1), A[2].flip(1)), -1),
        "torch: flip + permute + contiguous":
            lambda: A.flip(2).permute(1, 2, 3, 0).contiguous(),
        "torch: out.permute(3,...).copy_(flipped)":
            lambda: out.permute(3, 0, 1, 2).copy_(A.flip(2)),
        "numpy: stack of reversed views":
            lambda: np.stack((An[0, :, ::-1], An[1, :, ::-1], An[2, :, ::-1]), -1),
        "numpy: 3x strided assign into the output":
            np_assign,
        "torch: int32-packed surface, drop the 4th byte":
            lambda: v.flip(1)[..., :3].contiguous(),
        "[reference] contiguous copy of the same bytes":
            lambda: out.copy_(out),
    }
    doc = {"host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "threads": torch.get_num_threads(), "torch": torch.__version__,
           "shape": [B, H, W], "cases": {}}
    print(f"{doc['host']}  {doc['cpu']}  {doc['threads']} thread(s)  B={B} {H}x{W}\n")
    for name, fn in cases.items():
        fn()
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        dt = statistics.median(ts)
        doc["cases"][name] = {"ms_per_batch": dt * 1e3, "ms_per_frame": dt / B * 1e3,
                              "ms_min": min(ts) * 1e3, "ms_max": max(ts) * 1e3}
        print(f"{name:50s} {dt * 1e3:7.2f} ms/batch  {dt / B * 1e3:6.3f} ms/frame")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
