"""Paired before/after timing of one render hot path per family, plus ``step``, on CPU.

Step 9 is not an optimisation phase, but a hot path must not materially regress (plan §8): a
repeatable >10% regression blocks the merge unless it is explained and accepted. Same method as
``scratch/step9/foundation/pusht_perf.py`` — the two trees are measured by ALTERNATING subprocesses
(A B A B ...) so a thermal or scheduler drift lands on both arms instead of on whichever went second,
threads are pinned to one, and the reported number is the median of the per-repeat medians.

One cell per family, because within a family every cell shares the code the migration touched:

  * ``CartPole`` for classic control — ``gfxdraw`` per-primitive path, 400x600, the family's
    ``render`` is one function on ``_Classic``;
  * ``TwoRoom`` for navigation — the ``navdot`` torch path at 224x224, and the cell whose ``step``
    the ``_advance_rows`` refactor rewrote most.

What could have cost something: ``step`` now builds a StepResult (for classic control that means
reading the float64 ``_sim`` into a TensorDict; for navigation an ``unpack`` of the packed row), and
``render`` gained one ``isinstance`` on its argument. ``render`` is timed from PACKED ROWS, which is
what every corpus consumer and every certification check hands it.

The baseline arm needs a checkout of the Step-8 SHA; create one first (and remove it afterwards)::

    git worktree add --detach /tmp/se9a-baseline 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-classic-nav/perf.py
    git worktree remove /tmp/se9a-baseline
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

#: cell -> (num_envs, render iterations). Batches are the ones these renderers are used at on a CPU:
#: CartPole draws 400x600, which is 15x PushT's default frame, so 64 rather than 256.
CELLS = {"CartPole": 64, "TwoRoom": 256}

PROBE = r'''
import json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
import stable_envs as se

name, n, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cls = se.REGISTRY[name]
try:
    env = cls(num_envs=n, device="cpu")
except TypeError:
    env = cls(batch=n, device="cpu")
rng = np.random.default_rng(0)
states = np.asarray(env.sample_states(n, rng), np.float32)
env.set_state(states)
lo, hi = float(env.action_low), float(env.action_high)
action = torch.as_tensor(rng.uniform(lo, hi, (n, env.action_dim)).astype(np.float32))

def timed(fn, iters):
    for _ in range(3):
        fn()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out

st = torch.as_tensor(states)
render_ms = timed(lambda: env.render(st), iters)
step_ms = timed(lambda: env.step(action), iters)
print("@@JSON@@" + json.dumps({"render_ms": render_ms, "step_ms": step_ms}))
'''


def _run(tree: Path, python: str, cell: str, n: int, iters: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", PROBE, cell, str(n), str(iters)], cwd=str(tree),
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.splitlines()[-1][len("@@JSON@@"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="/tmp/se9a-baseline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out",
                    default=str(ROOT / "examples/results/step9/envs-classic-nav/performance.json"))
    a = ap.parse_args()

    arms = {"baseline": Path(a.baseline), "migrated": ROOT}
    out_cells = {}
    for cell, n in CELLS.items():
        per_repeat = {k: {"render_ms": [], "step_ms": []} for k in arms}
        for _ in range(a.repeats):
            for arm, tree in arms.items():
                got = _run(tree, a.python, cell, n, a.iters)
                for key in ("render_ms", "step_ms"):
                    per_repeat[arm][key].append(statistics.median(got[key]))
        summary = {}
        for key in ("render_ms", "step_ms"):
            b = statistics.median(per_repeat["baseline"][key])
            m = statistics.median(per_repeat["migrated"][key])
            summary[key] = {"baseline_median_ms": round(b, 4), "migrated_median_ms": round(m, 4),
                            "delta_pct": round(100.0 * (m - b) / b, 2),
                            "baseline_repeats": [round(x, 4) for x in per_repeat["baseline"][key]],
                            "migrated_repeats": [round(x, 4) for x in per_repeat["migrated"][key]]}
        out_cells[cell] = {"num_envs": n, "summary": summary}

    record = {"host": os.uname().nodename, "device": "cpu", "threads": 1,
              "iters": a.iters, "repeats": a.repeats,
              "paired": "alternating subprocesses, median of per-repeat medians",
              "python": a.python, "baseline_tree": str(a.baseline), "cells": out_cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: v["summary"] for c, v in out_cells.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
