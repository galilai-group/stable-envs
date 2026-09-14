"""The two experiments that decided RASTER2D-2's shape, kept so the numbers in findings.md resolve.

Both answer "what does the batch-union window cost, and what is the cheapest exact thing to replace
it with", on the real agent positions of the frozen Step 4 corpus rather than on a model.

  --mode chunk    split the batch into fixed chunks and take a window per chunk. This is the
                  obvious replacement and it is the one that was REJECTED: the gain only arrives at
                  a chunk of one, which is a Python loop over the batch.
  --mode group    quantise each row's box ORIGIN onto a grid of the box size and take a window per
                  cell. Flat in the batch, and what shipped.
  --mode pergroup what one group costs in dispatch, on a scene built so the pixel work is fixed and
                  only the group count varies. This is where _GROUP_DISPATCH_PIXELS comes from.

    python scratch/step6/raster2d/window_experiment.py --mode chunk
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import stable_envs as se                                             # noqa: E402
from stable_envs.render import raster2d as R2                        # noqa: E402
from stable_envs.render.scene import Scene2D                         # noqa: E402


def med(f, n=9, w=3):
    for _ in range(w):
        f()
    s = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s)


def chunk_mode(B: int) -> None:
    """The agent disc of SimplePointMaze, painted in chunks of the batch, timed.

    One primitive in isolation and not through the renderer, deliberately: the question is what a
    windowing strategy costs for a primitive whose position is per environment, and putting the rest
    of the scene beside it would only add a constant to every row of the table.
    """
    pool = np.load(ROOT / "examples/results/step4/states/SimplePointMaze.npz")["states"]
    st = torch.as_tensor(np.asarray(pool[np.arange(B) % len(pool)], np.float32))
    S, world, r = 500, 5.0, 0.1
    px = (torch.arange(S, dtype=torch.float32).view(1, S) + 0.5) * (world / S)
    py = world - (torch.arange(S, dtype=torch.float32).view(S, 1) + 0.5) * (world / S)
    PX, PY = px.expand(S, S), py.expand(S, S)
    canvas = torch.randint(0, 255, (B, S, S, 3), dtype=torch.uint8)
    col = torch.zeros(1, 1, 1, 3, dtype=torch.uint8)

    def window(rows):
        c = st[rows]
        s = world / S
        c0 = max(0, int(np.floor(float((c[:, 0] - r).min()) / s - 0.5)) - 1)
        c1 = min(S, int(np.ceil(float((c[:, 0] + r).max()) / s - 0.5)) + 2)
        r0 = max(0, int(np.floor((world - float((c[:, 1] + r).max())) / s - 0.5)) - 1)
        r1 = min(S, int(np.ceil((world - float((c[:, 1] - r).min())) / s - 0.5)) + 2)
        return r0, r1, c0, c1

    def run(C):
        for lo in range(0, B, C):
            hi = min(lo + C, B)
            r0, r1, c0, c1 = window(list(range(lo, hi)))
            if r1 <= r0 or c1 <= c0:
                continue
            n = hi - lo
            p, q = PX[r0:r1, c0:c1].unsqueeze(0), PY[r0:r1, c0:c1].unsqueeze(0)
            d2 = (p - st[lo:hi, 0].view(n, 1, 1)) ** 2 + (q - st[lo:hi, 1].view(n, 1, 1)) ** 2
            w = canvas[lo:hi, r0:r1, c0:c1, :]
            w.copy_(torch.where((d2 <= r * r).unsqueeze(-1), col, w))

    print(f"SimplePointMaze agent disc, {S} px, batch {B}, one thread. Chunk the BATCH:")
    for C in (B, B // 2, B // 4, 8, 4, 2, 1):
        if C < 1:
            continue
        print(f"   chunk {C:4d}: {med(lambda c=C: run(c)):8.2f} ms   first window {window(list(range(0, C)))}")
    print("   -> the gain arrives only at a chunk of one, which is a Python loop over the batch and"
          " O(B) dispatch. Rejected; see --mode group.")


def group_mode() -> None:
    """What the shipped grouping does to the window work, per primitive, on the frozen states."""
    print("union vs grouped window work, in frame-equivalents (1.0 = one full-frame pass per row):")
    for cfg in ("PushT", "SimplePointMaze"):
        pool = np.load(ROOT / f"examples/results/step4/states/{cfg}.npz")["states"]
        for B in (64, 256, 1024):
            st = torch.as_tensor(np.asarray(pool[np.arange(B) % len(pool)], np.float32))
            e = se.make(cfg, batch=B, device="cpu")
            sc = e.scene(st)
            S = int(e.size) * int(e.ss)
            saved = R2._GROUP_DISPATCH_PIXELS
            try:
                R2._GROUP_DISPATCH_PIXELS = 10 ** 12
                off = R2._prim_plan(sc, S, S)
                R2._GROUP_DISPATCH_PIXELS = 0
                on = R2._prim_plan(sc, S, S)
            finally:
                R2._GROUP_DISPATCH_PIXELS = saved
            live = R2._prim_plan(sc, S, S)
            wpx = lambda pl: sum(sum((b - a) * (d - c) * (B if r is None else len(r))     # noqa: E731
                                     for r, a, b, c, d in x[6]) for x in pl) / (S * S)
            print(f"   {cfg:17s} B={B:<5} union {wpx(off):9.2f}   grouped {wpx(on):9.2f}   "
                  f"shipped {wpx(live):9.2f}   groups {sum(len(x[6]) for x in live)}")


def pergroup_mode() -> None:
    """What one group costs in dispatch: B discs in B distinct cells, so the pixel work is fixed."""
    S = 64
    print("one group's dispatch cost, from a scene whose grouped pixel work does not change with B:")
    for B in (8, 32, 128, 512):
        gen = torch.Generator().manual_seed(1)
        g = torch.zeros(B, 1, 8)
        g[:, 0, 0:2] = torch.rand(B, 2, generator=gen) * 0.9 + 0.05
        g[:, 0, 2], g[:, 0, 3] = 0.02, -1.0
        sc = Scene2D(kind=torch.ones(B, 1, dtype=torch.int32), geom=g,
                     fill=torch.full((B, 1, 3), 200.0), line=torch.full((B, 1, 3), 200.0),
                     line_w=torch.zeros(B, 1), background=torch.zeros(B, 3),
                     world=(0.0, 0.0, 1.0, 1.0)).validate()
        saved = R2._GROUP_DISPATCH_PIXELS
        try:
            R2._GROUP_DISPATCH_PIXELS = 10 ** 12
            off = med(lambda: R2.render_torch(sc, S, 1))
            a = R2.render_torch(sc, S, 1).clone()
            R2._GROUP_DISPATCH_PIXELS = 0
            on = med(lambda: R2.render_torch(sc, S, 1))
            b = R2.render_torch(sc, S, 1)
            plan = R2._prim_plan(sc, S, S)
        finally:
            R2._GROUP_DISPATCH_PIXELS = saved
        assert torch.equal(a, b), "the two windowings drew different pictures"
        G = len(plan[0][6])
        wpx = sum((b_ - a_) * (d - c) * len(r) for r, a_, b_, c, d in plan[0][6])
        empty = Scene2D(kind=torch.zeros(B, 1, dtype=torch.int32), geom=g,
                        fill=torch.full((B, 1, 3), 200.0), line=torch.full((B, 1, 3), 200.0),
                        line_w=torch.zeros(B, 1), background=torch.zeros(B, 3),
                        world=(0.0, 0.0, 1.0, 1.0)).validate()
        base = med(lambda: R2.render_torch(empty, S, 1))
        per = (on - base - wpx * 6e-6) / max(G, 1) * 1e3
        print(f"   B={B:<4} groups {G:<4} ungrouped {off:8.3f} ms  grouped {on:8.3f} ms  "
              f"canvas only {base:7.3f} ms  ->  {per:6.1f} us per group")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("chunk", "group", "pergroup"), default="chunk")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"host torch {torch.__version__}, {torch.get_num_threads()} thread(s), "
          f"tree {se.__file__}")
    {"chunk": lambda: chunk_mode(a.batch), "group": group_mode, "pergroup": pergroup_mode}[a.mode]()


if __name__ == "__main__":
    main()
