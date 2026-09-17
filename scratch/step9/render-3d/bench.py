"""Paired, alternating before/after timings of the 3-D render hot path (plan §8, §21).

WHY ALTERNATING. A "before" block followed by an "after" block measures the machine's mood as much
as the code: a thermal ramp, another tenant arriving on the node, or a CPU frequency step lands
entirely on one leg. So the legs are interleaved -- baseline, head, baseline, head, ... -- in
separate interpreters, and the statistic is the median over rounds of each leg. The two legs are the
same venv, the same host, the same process shape and the same states; the only difference is which
tree is on ``PYTHONPATH``.

WHAT IS TIMED is the PUBLIC caller-visible path (``env.render()`` on a state that is already set),
not a private kernel, because that is where a structural change would show up and it is what a
consumer pays. The configurations are the cells that reach ``raycast3d`` and ``cpu3d`` at the sizes
and batch shapes their Step-7 tiers use.

The bar (plan §8): a repeatable regression of roughly >10 % on a hot path blocks merge. ``ratio`` is
head/baseline wall time, so >1.10 is a regression and the per-leg spread printed beside it says
whether the machine could resolve it.

Usage::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/render-3d/bench.py            # CPU
    python scratch/step9/render-3d/bench.py --device cuda --set gpu                             # in the job
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: ``(config, batch)``. CPU: the cells whose render is raycast3d's analytic cast over a baked plate
#: and the four trackcom cells cpu3d owns, at their registered sizes (84 / 224 / 240 px).
SETS: dict[str, list[tuple[str, int]]] = {
    "cpu": [
        ("DMCReacherHard", 64),          # raycast3d, 4 primitives, 84 px -- the cheapest cast
        ("DMCBallInCupCatch", 64),       # raycast3d, 36 primitives, 84 px -- the dearest
        ("DMCFingerTurnEasy", 64),       # raycast3d, 84 px, the cell the tensor caches were found on
        ("DMCManipulatorBringBall", 16),  # raycast3d, 224 px
        ("MujocoPusher", 16),            # raycast3d, 240 px
        ("DMCWalkerWalk", 64),           # cpu3d, 84 px
        ("DMCHumanoidStand", 64),        # cpu3d, 84 px, the most movers
        ("DMCQuadrupedWalk", 16),        # cpu3d, 224 px, the ELLIPSOID torso
    ],
    #: GPU: the same raycast3d cells plus the OGBench mazes, which are where the fused Triton cast
    #: and the tiled composite actually run.
    "gpu": [
        ("DMCReacherHard", 256),
        ("DMCBallInCupCatch", 256),
        ("DMCManipulatorBringBall", 64),
        ("PointmazeLargeNavigate", 256),
        ("AntmazeLargeExplore", 256),
        ("HumanoidmazeGiantNavigate", 64),
        ("OGBCube", 64),
        ("DMCWalkerWalk", 256),
    ],
}

LEG = r'''
import json, sys, time
import numpy as np, torch
import stable_envs as se

name, batch, device, warmup, reps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
corpus = sys.argv[6] or None
cls = se.REGISTRY[name]
env = cls(batch=batch, device=device)
if corpus:
    s = np.load(corpus)["states"].astype(np.float32)
    s = np.resize(s, (batch, s.shape[1]))
else:
    s = np.asarray(env.sample_states(batch, np.random.default_rng(0)), np.float32)
env.set_state(s)
sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
for _ in range(warmup):
    env.render(); sync()
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); env.render(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
print("@@JSON@@" + json.dumps({"ms": ts, "size": int(env.size)}))
'''


def _worktree(sha: str) -> tuple[Path, bool]:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-render3d-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-render3d-baseline-{sha[:12]}"
    if path.exists():
        return path, False
    subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                   check=True, capture_output=True)
    return path, True


def _leg(tree: Path, python: str, name: str, batch: int, device: str, warmup: int, reps: int,
         corpus: str | None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    r = subprocess.run([python, "-c", LEG, name, str(batch), device, str(warmup), str(reps),
                        str(tree / corpus) if corpus else ""],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        return None, f"{r.stdout[-600:]}\n{r.stderr[-600:]}"
    return json.loads(hit[-1][len("@@JSON@@"):]), None


#: Frozen states where one exists, so the timed frame is the certified frame and not a random pose.
CORPUS = {n: f"examples/results/step4/states/{n}.npz" for n in (
    "DMCReacherHard", "DMCBallInCupCatch", "DMCFingerTurnEasy", "DMCWalkerWalk",
    "DMCHumanoidStand", "PointmazeLargeNavigate", "AntmazeLargeExplore",
    "HumanoidmazeGiantNavigate")}
CORPUS["DMCManipulatorBringBall"] = \
    "examples/results/step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz"
CORPUS["DMCQuadrupedWalk"] = \
    "examples/results/step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz"
CORPUS["OGBCube"] = "examples/results/step7/ogb-manipulation/states/OGBCube.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", dest="which", default="cpu", choices=sorted(SETS))
    ap.add_argument("--rounds", type=int, default=5, help="alternating before/after rounds")
    ap.add_argument("--reps", type=int, default=5, help="timed renders per leg")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-3d"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    baseline, created = _worktree(a.baseline_sha)
    try:
        rows = []
        for name, batch in SETS[a.which]:
            legs: dict[str, list[float]] = {"baseline": [], "head": []}
            err = None
            for _ in range(a.rounds):
                for label, tree in (("baseline", baseline), ("head", ROOT)):
                    got, e = _leg(tree, a.python, name, batch, a.device, a.warmup, a.reps,
                                  CORPUS.get(name))
                    if got is None:
                        err = e
                        break
                    legs[label].append(statistics.median(got["ms"]))
                if err:
                    break
            if err:
                rows.append({"config": name, "batch": batch, "status": "skipped",
                             "reason": err.strip().splitlines()[-1][:160] if err.strip() else "?"})
                continue
            b = statistics.median(legs["baseline"])
            h = statistics.median(legs["head"])
            spread = (max(legs["baseline"]) - min(legs["baseline"])) / b if b else 0.0
            rows.append({"config": name, "batch": batch, "status": "ok",
                         "baseline_ms": round(b, 3), "head_ms": round(h, 3),
                         "ratio_head_over_baseline": round(h / b, 4),
                         "baseline_leg_spread": round(spread, 4),
                         "baseline_rounds_ms": [round(x, 3) for x in legs["baseline"]],
                         "head_rounds_ms": [round(x, 3) for x in legs["head"]]})
            print(f"{name:28s} B={batch:<5d} baseline {b:9.2f} ms  head {h:9.2f} ms  "
                  f"ratio {h / b:5.3f}  (baseline spread {spread:5.1%})")
        worst = max((r["ratio_head_over_baseline"] for r in rows if r["status"] == "ok"),
                    default=0.0)
        record = {"baseline_sha": a.baseline_sha, "device": a.device, "set": a.which,
                  "host": os.uname().nodename, "python": a.python, "rounds": a.rounds,
                  "reps_per_leg": a.reps, "rows": rows, "worst_ratio": worst,
                  "regression_bar": 1.10, "blocks_merge": bool(worst > 1.10),
                  "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                             capture_output=True, text=True).stdout.strip()}
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"timings_{a.tag or a.which}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out / fn}\nworst ratio {worst:.3f} against a 1.10 bar: "
              f"{'REGRESSION' if worst > 1.10 else 'within bar'}")
        return 0
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
