"""Paired before/after timing of one render hot path per BACKEND PATH, on CPU.

Step 9 is not an optimisation phase, but a hot path must not materially regress (plan §8): a
repeatable >10% regression blocks the merge unless it is explained and accepted. Same method as
``scratch/step9/envs-classic-nav/perf.py`` — the two trees are measured by ALTERNATING subprocesses
(A B A B ...) so a thermal or scheduler drift lands on both arms instead of on whichever went second,
threads are pinned to one, and the reported number is the median of the per-repeat medians.

One cell per backend path, because within a path every cell runs the same code:

  * ``DMCCartpoleBalance`` — the PARITY path: three baked plates and ``raycast3d``'s analytic mover
    pass at the shipped 84 px / ss=1. The plate bake is cached per (domain, size, appearance), so
    what is timed is the per-frame half, which is what a corpus consumer pays;
  * ``DMCCheetahRun`` — the VISUAL path on CPU: ``cpu3d``'s raytrace of the whole scene every frame,
    which is the only trackcom backend a machine without CUDA can run (MJWarp is the CUDA arm and is
    timed in ``gpu.sbatch``).

``step`` is NOT timed: no DM Control cell in this package has one — they ship a renderer over
dm_control's own physics and refuse ``step``/``transition`` by declaration — so there is no dynamics
hot path to regress. ``render`` is timed from PACKED ROWS, which is what every corpus consumer and
every certification check hands it.

What could have cost something: ``render`` gained one ``isinstance`` on its argument through the
foundation's guard, and the module became a package, so the constructor now resolves its caches
through a ``from ... import``ed dict instead of a module global. Both are per-call constants against
a per-frame ray cast, and the measurement is what says so.

The baseline arm needs a checkout of the Step-8 SHA; create one first (and remove it afterwards)::

    git worktree add --detach /tmp/se9b-baseline 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc
    MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python scratch/step9/envs-dmcontrol/perf.py
    git worktree remove /tmp/se9b-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: cell -> num_envs. 16 rather than 64: the CPU trackcom path ray-casts the whole scene per frame and
#: a 64-env repeat would put one measurement over a minute, while the per-frame cost is linear in the
#: batch and the ratio this test is about is not.
CELLS = {"DMCCartpoleBalance": 16, "DMCCheetahRun": 16}

PROBE = r'''
import json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
import stable_envs as se

name, n, iters, dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cls = se.REGISTRY[name]
try:
    env = cls(num_envs=n, device=dev)
except TypeError:
    env = cls(batch=n, device=dev)
rng = np.random.default_rng(0)
states = np.asarray(env.sample_states(n, rng), np.float32)
env.set_state(states)

def timed(fn, iters):
    """Wall time per call, in ms, with the device drained first and after each call.

    ``torch.cuda.synchronize`` is what makes a CUDA number a number: without it the loop measures
    how fast the host can enqueue work.
    """
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    for _ in range(3):
        fn()
    sync()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t) * 1e3)
    return out

st = torch.as_tensor(states)
render_ms = timed(lambda: env.render(st), iters)
print("@@JSON@@" + json.dumps({"render_ms": render_ms}))
'''


def _run(tree: Path, python: str, cell: str, n: int, iters: int, dev: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", PROBE, cell, str(n), str(iters), dev], cwd=str(tree),
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.splitlines()[-1][len("@@JSON@@"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="/tmp/se9b-baseline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    #: Which cells to time. The default is one per CPU backend path; the GPU job passes two parity
    #: cells and one trackcom cell, because on CUDA those are the fused mover pass and MJWarp.
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out",
                    default=str(ROOT / "examples/results/step9/envs-dmcontrol/performance.json"))
    a = ap.parse_args()

    arms = {"baseline": Path(a.baseline), "migrated": ROOT}
    cells = {c: CELLS.get(c, 16) for c in a.cells.split(",") if c}
    out_cells = {}
    for cell, n in cells.items():
        per_repeat = {k: {"render_ms": []} for k in arms}
        for _ in range(a.repeats):
            for arm, tree in arms.items():
                got = _run(tree, a.python, cell, n, a.iters, a.device)
                for key in ("render_ms",):
                    per_repeat[arm][key].append(statistics.median(got[key]))
        summary = {}
        for key in ("render_ms",):
            b = statistics.median(per_repeat["baseline"][key])
            m = statistics.median(per_repeat["migrated"][key])
            summary[key] = {"baseline_median_ms": round(b, 4), "migrated_median_ms": round(m, 4),
                            "delta_pct": round(100.0 * (m - b) / b, 2),
                            "baseline_repeats": [round(x, 4) for x in per_repeat["baseline"][key]],
                            "migrated_repeats": [round(x, 4) for x in per_repeat["migrated"][key]]}
        out_cells[cell] = {"num_envs": n, "summary": summary}

    record = {"host": os.uname().nodename, "device": a.device, "threads": 1,
              "iters": a.iters, "repeats": a.repeats,
              "paired": "alternating subprocesses, median of per-repeat medians",
              "python": a.python, "baseline_tree": str(a.baseline), "cells": out_cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: v["summary"] for c, v in out_cells.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
