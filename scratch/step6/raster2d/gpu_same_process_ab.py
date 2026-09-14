"""Is the cuda path the same speed as the path it replaced? Asked in ONE process, interleaved.

Two SLURM legs cannot answer this. A GPU render at batch 1 is host-launch bound, the two legs run
minutes apart on a shared node, and the spread WITHIN one leg's two runs came out the same size as
the difference between the legs -- PiecewiseZones read 1,031 and 1,075 f/s for the same tree. Any
statistic over two cross-process samples is then reporting drift.

So this times both bindings in one process, interleaved A/B/A/B on one warmed card, and reports the
paired median ratio. ``swm_nav``'s two Step 6 methods take the branch they take from
``self.device.type``; binding them to the expressions they replaced is exactly the pre-SWMNAV-1
render, so a ratio of 1.00 here is the claim "on cuda this is the same code" measured rather than
read off the diff.

    python scratch/step6/raster2d/gpu_same_process_ab.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import stable_envs as se                                        # noqa: E402
from stable_envs.envs import swm_nav as SN                      # noqa: E402

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


def med(fn, device, reps, warmup):
    cuda = device.startswith("cuda")
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved A/B rounds; best of each")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    note = {"device": a.device, "stable_envs_file": se.__file__, "batches": a.batch,
            "reps": a.reps, "rounds": a.rounds, "rows": []}
    if a.device.startswith("cuda"):
        note["card"] = torch.cuda.get_device_name(0)
    print(f"tree {se.__file__}  device {a.device}  {note.get('card', '')}")
    print(f"{'config':17s}{'B':>7}{'pre ms':>10}{'now ms':>10}{'ratio':>8}   pixels")
    worst = (9e9, None)
    for cfg in CONFIGS:
        pool = np.load(ROOT / f"examples/results/step4/states/{cfg}.npz")["states"]
        for B in a.batch:
            st = np.asarray(pool[np.arange(B) % len(pool)], np.float32)
            e = se.make(cfg, batch=B, device=a.device)
            new_base, new_dot = e._base_image, e._blend_dot

            def old_base(sty, lo, hi, _e=e):
                return _e.base_image(lambda p: sty[p][lo:hi], hi - lo)

            def old_dot(img, pos, sig, col, _e=e):
                return SN._alpha_blend(img, SN._gaussian_dot(_e._gy, _e._gx, pos, sig), col)

            def bind(base, dot):
                e._base_image, e._blend_dot = base, dot

            bind(old_base, old_dot)
            ref = e.render(st).clone()
            bind(new_base, new_dot)
            same = bool(torch.equal(ref, e.render(st)))
            pre = now = 1e18
            for _ in range(a.rounds):                      # interleaved, best of each binding
                bind(old_base, old_dot)
                pre = min(pre, med(lambda: e.render(st), a.device, a.reps, a.warmup))
                bind(new_base, new_dot)
                now = min(now, med(lambda: e.render(st), a.device, a.reps, a.warmup))
            r = pre / now
            if r < worst[0]:
                worst = (r, (cfg, B))
            note["rows"].append({"config": cfg, "batch": B, "pre_ms": pre, "now_ms": now,
                                 "ratio": r, "pixels_identical": same})
            print(f"{cfg:17s}{B:>7}{pre:10.3f}{now:10.3f}{r:8.3f}   "
                  f"{'identical' if same else 'DIFFER'}", flush=True)
            assert same, f"{cfg} B={B}: the cuda branch is not drawing what it replaced"
            del e
    print(f"\nworst {worst[0]:.3f} at {worst[1]}")
    note["worst"] = {"ratio": worst[0], "config": worst[1][0], "batch": worst[1][1]}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(note, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
