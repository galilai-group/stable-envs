"""Paired before/after CPU timing of the Craftax tilemap render (plan §8).

Step 9 is not an optimisation phase, but a hot path must not materially regress: a repeatable >10%
regression blocks the merge unless it is causally explained and accepted. Same method as
``scratch/step9/foundation/pusht_perf.py`` and ``scratch/step9/envs-classic-nav/perf.py`` — the two
trees are measured by ALTERNATING subprocesses (A B A B ...) so a thermal or scheduler drift lands on
both arms instead of on whichever went second, torch threads and OMP/MKL are pinned to one, and the
reported number is the median of the per-repeat medians.

WHAT IS TIMED, and why it is timed twice per cell. Craftax's render has two paths and the migration
had to leave both alone (module docstring):

  * the **uint8 gather** — a daylight, awake, unit-gain batch is one ``index_select`` from a uint8
    atlas with no float buffer anywhere;
  * the **float chain** — a sleeping or lit batch gathers the pre-rounding float table and pays the
    reference's per-pixel post-process, which is about 100x the gather on one core.

A single mixed batch would average them and hide a regression in either, so each is timed on a batch
drawn entirely from its own band. ``render`` is timed FROM PACKED ROWS, which is what every corpus
consumer and every certification check hands it.

``PFRocketLanding`` is absent because it does not render on this host at all (nvdiffrast is CUDA-only
and the CPU request is refused by design); its timing is owed to the GPU oracle.

    git worktree add --detach /tmp/se9e-baseline 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc
    MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python scratch/step9/envs-craftax-rocket/perf.py
    git worktree remove /tmp/se9e-baseline
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

#: cell -> (num_envs, tile size). The registered default size for each, at a batch a CPU consumer
#: would actually use.
CELLS = {"CraftaxClassic": (64, 16), "CraftaxFull": (64, 10)}

PROBE = r'''
import json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
import stable_envs as se

name, n, size, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cls = se.REGISTRY[name]
try:
    env = cls(num_envs=n, device="cpu")
except TypeError:
    env = cls(batch=n, device="cpu")
rng = np.random.default_rng(0)
states = np.asarray(env.sample_states(n, rng), np.float32)

names = list(env.state_names)
LIGHT, SLEEP = names.index("light_level"), names.index("is_sleeping")

# The uint8 fast path is defined by the STATE, not by the call: fully lit, awake, unit gain. The float
# chain is everything else; a sleeping row takes it whatever its lighting is.
fast = states.copy()
fast[:, LIGHT] = 1.0
fast[:, SLEEP] = 0.0
if name == "CraftaxFull":
    lm = [i for i, nm in enumerate(names) if nm.startswith("light_r")]
    fast[:, lm] = 1.0
slow = states.copy()
slow[:, SLEEP] = 1.0
slow[:, LIGHT] = 0.4

def timed(fn, iters):
    for _ in range(3):
        fn()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out

f, s = torch.as_tensor(fast), torch.as_tensor(slow)
print("@@JSON@@" + json.dumps({
    "uint8_gather_ms": timed(lambda: env.render(f, size=size), iters),
    "float_chain_ms": timed(lambda: env.render(s, size=size), iters),
}))
'''

KEYS = ("uint8_gather_ms", "float_chain_ms")


def _run(tree: Path, python: str, cell: str, n: int, size: int, iters: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", PROBE, cell, str(n), str(size), str(iters)], cwd=str(tree),
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.splitlines()[-1][len("@@JSON@@"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="/tmp/se9e-baseline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out",
                    default=str(ROOT / "examples/results/step9/envs-craftax-rocket/performance.json"))
    a = ap.parse_args()

    arms = {"baseline": Path(a.baseline), "migrated": ROOT}
    out_cells = {}
    for cell, (n, size) in CELLS.items():
        per_repeat = {k: {key: [] for key in KEYS} for k in arms}
        for _ in range(a.repeats):
            for arm, tree in arms.items():
                got = _run(tree, a.python, cell, n, size, a.iters)
                for key in KEYS:
                    per_repeat[arm][key].append(statistics.median(got[key]))
        summary = {}
        for key in KEYS:
            b = statistics.median(per_repeat["baseline"][key])
            m = statistics.median(per_repeat["migrated"][key])
            summary[key] = {"baseline_median_ms": round(b, 4), "migrated_median_ms": round(m, 4),
                            "delta_pct": round(100.0 * (m - b) / b, 2),
                            "baseline_repeats": [round(x, 4) for x in per_repeat["baseline"][key]],
                            "migrated_repeats": [round(x, 4) for x in per_repeat["migrated"][key]]}
        out_cells[cell] = {"num_envs": n, "tile_px": size, "summary": summary}

    record = {"host": os.uname().nodename, "device": "cpu", "threads": 1,
              "iters": a.iters, "repeats": a.repeats,
              "paired": "alternating subprocesses, median of per-repeat medians",
              "python": a.python, "baseline_tree": str(a.baseline),
              "owed": {"PFRocketLanding": "renders only on cuda; timing owed to the GPU oracle"},
              "cells": out_cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: v["summary"] for c, v in out_cells.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
