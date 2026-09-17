"""Paired before/after timing of PushT's two hot paths on CPU: ``step`` and ``render``.

Step 9 is not an optimisation phase, but a hot path must not materially regress (plan §8): a
repeatable >10% regression blocks the merge unless it is explained and accepted. The two paths the
Wave-0 migration could plausibly cost something are the ones measured here:

  * ``step`` now BUILDS A STEPRESULT — it unpacks the packed row into a TensorDict and splices in the
    agent velocity — where it used to return the ``(B, 5)`` buffer itself;
  * ``render`` gained one ``isinstance`` on its argument.

The trees are measured by ALTERNATING subprocesses (A B A B ...) rather than one after the other, so a
thermal or scheduler drift during the run lands on both arms instead of on whichever went second, and
the reported number is the median of the per-repeat medians. Threads are pinned to one, as every CPU
measurement in this repository is.

The baseline arm needs a checkout of the Step-8 SHA; create one first (and remove it afterwards)::

    git worktree add --detach /tmp/se9-baseline 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/foundation/pusht_perf.py
    git worktree remove /tmp/se9-baseline
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

PROBE = r'''
import json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
import stable_envs as se

n, iters, size = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cls = se.REGISTRY["PushT"]
try:
    env = cls(num_envs=n, device="cpu", size=size)
except TypeError:
    env = cls(batch=n, device="cpu", size=size)
rng = np.random.default_rng(0)
states = np.asarray(env.sample_states(n, rng), np.float32)
env.set_state(states)
action = torch.as_tensor(rng.uniform(60, 452, (n, 2)).astype(np.float32))

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


def _run(tree: Path, python: str, n: int, iters: int, size: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", PROBE, str(n), str(iters), str(size)], cwd=str(tree),
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.splitlines()[-1][len("@@JSON@@"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="/tmp/se9-baseline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/foundation/performance.json"))
    a = ap.parse_args()

    arms = {"baseline": Path(a.baseline), "migrated": ROOT}
    per_repeat = {k: {"render_ms": [], "step_ms": []} for k in arms}
    for _ in range(a.repeats):
        for name, tree in arms.items():
            got = _run(tree, a.python, a.num_envs, a.iters, a.size)
            for key in ("render_ms", "step_ms"):
                per_repeat[name][key].append(statistics.median(got[key]))

    summary = {}
    for key in ("render_ms", "step_ms"):
        b = statistics.median(per_repeat["baseline"][key])
        m = statistics.median(per_repeat["migrated"][key])
        summary[key] = {"baseline_median_ms": round(b, 4), "migrated_median_ms": round(m, 4),
                        "delta_pct": round(100.0 * (m - b) / b, 2),
                        "baseline_repeats": [round(x, 4) for x in per_repeat["baseline"][key]],
                        "migrated_repeats": [round(x, 4) for x in per_repeat["migrated"][key]]}
    record = {"host": os.uname().nodename, "device": "cpu", "threads": 1,
              "num_envs": a.num_envs, "size": a.size, "iters": a.iters, "repeats": a.repeats,
              "paired": "alternating subprocesses, median of per-repeat medians",
              "python": a.python, "baseline_tree": str(a.baseline), "summary": summary}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
