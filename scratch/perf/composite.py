"""What the per-chunk COMPOSITE costs, op by op, against what its memory traffic says it should.

After the hoisting pass the remaining non-kernel time is all here: the channel swap, the mover mask and
the ``where``s that put MJWarp's movers on MuJoCo's plate. Each is timed alone at the shape a chunk
really has, next to the bytes it must move, so "torch is slow at this" is a ratio rather than a guess.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch


def timed(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--ngeom", type=int, default=40)
    a = ap.parse_args()
    n, r = a.n, a.res
    dev = "cuda"
    bgra = torch.randint(0, 255, (n, r, r, 4), dtype=torch.uint8, device=dev)
    seg = torch.randint(-1, a.ngeom, (n, r, r), dtype=torch.int32, device=dev)
    off = torch.randint(0, 255, (n, r, r, 3), dtype=torch.uint8, device=dev)
    img = torch.randint(0, 255, (8, r, r, 3), dtype=torch.uint8, device=dev)
    dark = torch.randint(0, 255, (8, r, r, 3), dtype=torch.uint8, device=dev)
    g = torch.randint(0, 8, (n,), dtype=torch.long, device=dev)
    lut = torch.zeros(a.ngeom + 2, dtype=torch.bool, device=dev)
    lut[3:12] = True
    movers = torch.arange(2, 11, device=dev, dtype=torch.int32)
    out = torch.empty((n, r, r, 3), dtype=torch.uint8, device=dev)

    frame_mb = n * r * r * 3 / 1e6
    res = {}

    def rec(label, fn, mb):
        dt = timed(fn)
        res[label] = {"ms": dt * 1e3, "GBs": mb / 1e3 / dt}
        print(f"{label:38s} {dt*1e3:8.3f} ms   {mb/1e3/dt:8.1f} GB/s effective", flush=True)

    rec("bgra[..., [2,1,0]] (advanced index)", lambda: bgra[:n][..., [2, 1, 0]],
        frame_mb * (4 / 3 + 1))
    rec("bgra[..., :3].flip(-1)", lambda: bgra[:n][..., :3].flip(-1), frame_mb * 2)
    rec("out.copy_(bgra[..., [2,1,0]])", lambda: out.copy_(bgra[..., [2, 1, 0]]),
        frame_mb * (4 / 3 + 2))
    rec("isin(seg, movers)", lambda: torch.isin(seg, movers), frame_mb * 4 / 3 * 2)
    rec("lut.index_select(seg+1)",
        lambda: lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape), frame_mb * 4 / 3 * 2)
    mv = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
    rgb = bgra[..., [2, 1, 0]].contiguous()
    rec("where(mv[...,None], rgb, img[g])",
        lambda: torch.where(mv.unsqueeze(-1), rgb, img[g]), frame_mb * 3)
    rec("(rgb != off).any(-1, keepdim=True)",
        lambda: (rgb != off).any(-1, keepdim=True), frame_mb * 2)
    rec("img[g] (gather)", lambda: img[g], frame_mb * 2)

    def fetch_composite():
        m = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
        c = torch.where(m.unsqueeze(-1), bgra[..., [2, 1, 0]], img[g])
        out.copy_(c)
    rec("FETCH composite (as shipped)", fetch_composite, frame_mb * 8)

    def mw_composite():
        m = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
        rr = bgra[..., [2, 1, 0]]
        pl = torch.where((rr != off).any(-1, keepdim=True) & ~m.unsqueeze(-1), dark[g], img[g])
        out.copy_(torch.where(m.unsqueeze(-1), rr, pl))
    rec("METAWORLD composite (as shipped)", mw_composite, frame_mb * 14)

    print("JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
