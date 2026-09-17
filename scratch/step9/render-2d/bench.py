"""Paired, alternating before/after timings of the 2-D render hot paths (plan §8, §21).

WHY ALTERNATING. A "before" block followed by an "after" block measures the machine's mood as much
as the code: a thermal ramp, another tenant arriving on the node, or a CPU frequency step lands
entirely on one leg. So the legs are interleaved -- baseline, head, baseline, head, ... -- in
separate interpreters, and the statistic is the median over rounds of each leg. The two legs are the
same venv, the same host, the same process shape and the same states; the only difference is which
tree is on ``PYTHONPATH``.

WHAT IS TIMED is the PUBLIC caller-visible path (``env.render()`` on a state that is already set),
not a private kernel, because that is where a structural change would show up and it is what a
consumer pays.

WHICH CELLS. The four hot paths the brief names, one cell each plus the second raster2d shape:

    raster2d   PushT (96 px, 9 primitives, the static-bake path)
               TwoRoom (224 px, the swm_nav base image and dot windows)
               SimplePointMaze (500 px) and Roundabout (600 px, 436 primitives) on the CPU set,
               because a per-primitive cost and a per-pixel cost are different mistakes
    gfxdraw    CartPole (400x600, the per-primitive Wu/scanline path this workstream consolidated)
               Acrobot and Pendulum, which are the other two shapes of the same path
    tilemap    Crafter (the atlas blit)
    prng       CraftaxClassic (144 px) and CraftaxFull

The bar (plan §8): a repeatable regression of roughly >10 % on a hot path blocks merge. ``ratio`` is
head/baseline wall time, so >1.10 is a regression and the per-leg spread printed beside it says
whether the machine could resolve it.

Usage::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/render-2d/bench.py         # CPU
    python scratch/step9/render-2d/bench.py --device cuda --set gpu                          # in the job
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

#: ``(config, batch)`` at the registered frame size. CPU batches are the Step-7 tier shapes, capped
#: so a 600 px torch raster finishes: a ratio is a ratio at any batch, and the batch is recorded.
SETS: dict[str, list[tuple[str, int]]] = {
    "cpu": [
        ("PushT", 64),                 # raster2d, 96 px, the static bake
        ("SimplePointMaze", 16),       # raster2d, 500 px
        ("Roundabout", 8),             # raster2d, 600 px, 436 primitives -- the dispatch-bound one
        ("TwoRoom", 32),               # navdot window + raster2d base, 224 px
        ("CartPole", 16),              # gfxdraw per-primitive, 400x600
        ("Acrobot", 16),               # gfxdraw, 500 px
        ("Pendulum", 16),              # gfxdraw, 500 px, the sprite blit
        ("Crafter", 32),               # tilemap blit, 64 px
        ("CraftaxClassic", 32),        # prng + atlas, 144 px
        ("CraftaxFull", 32),           # prng + atlas, 130x110
    ],
    #: GPU: the same cells at the batch the fused kernels are meant for, which is where the binned
    #: scene kernel, gfxdraw_classic's kernel, navdot's two passes and the tile blitter actually run.
    "gpu": [
        ("PushT", 1024),
        ("SimplePointMaze", 256),
        ("Roundabout", 256),
        ("TwoRoom", 256),
        ("TwoRoomTarget", 256),
        ("CartPole", 256),
        ("Acrobot", 256),
        ("Pendulum", 256),
        ("Crafter", 256),
        ("CraftaxClassic", 256),
        ("CraftaxFull", 256),
        ("CarRacing", 64),             # gfxdraw_fused, 1000x800 -- absent from the Mac entirely
        ("LunarLander", 256),
        ("MiniGridEmpty", 256),
        ("Sokoban", 256),
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

#: Frozen states where one exists, so the timed frame is the certified frame and not a random pose.
CORPUS = {n: f"examples/results/step4/states/{n}.npz" for n in (
    "PushT", "SimplePointMaze", "TwoRoom", "TwoRoomTarget", "CartPole", "Acrobot", "Pendulum")}
CORPUS["CraftaxClassic"] = "examples/results/step7/craftax-classic-full/corpus/states.npz"
CORPUS["CraftaxFull"] = "examples/results/step7/craftax-full/corpus/states.npz"


def _worktree(sha: str) -> tuple[Path, bool]:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-render2d-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-render2d-baseline-{sha[:12]}"
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
    cp = str(tree / corpus) if corpus and (tree / corpus).exists() else ""
    r = subprocess.run([python, "-c", LEG, name, str(batch), device, str(warmup), str(reps), cp],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        return None, f"{r.stdout[-600:]}\n{r.stderr[-600:]}"
    return json.loads(hit[-1][len("@@JSON@@"):]), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", dest="which", default="cpu", choices=sorted(SETS))
    ap.add_argument("--rounds", type=int, default=5, help="alternating before/after rounds")
    ap.add_argument("--reps", type=int, default=5, help="timed renders per leg")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-2d"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--arms", default="baseline,head",
                    help="which TREE each leg runs. ``baseline,baseline`` times the SAME tree "
                         "against itself in the same alternating shape, which is how a ratio that "
                         "is a property of the machine is told apart from one that is a property "
                         "of the patch. A same-tree run is not expected to give exactly 1.000; what "
                         "it gives is the floor under which a ratio means nothing.")
    a = ap.parse_args()
    arm_names = a.arms.split(",")
    if len(arm_names) != 2 or any(x not in ("baseline", "head") for x in arm_names):
        raise SystemExit("--arms takes exactly two of baseline|head")

    baseline, created = _worktree(a.baseline_sha)
    try:
        rows = []
        for name, batch in SETS[a.which]:
            legs: dict[str, list[float]] = {"baseline": [], "head": []}
            err = None
            for _ in range(a.rounds):
                trees = {"baseline": baseline, "head": ROOT}
                for label, arm in (("baseline", arm_names[0]), ("head", arm_names[1])):
                    tree = trees[arm]
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
                print(f"{name:22s} B={batch:<5d} skipped: {rows[-1]['reason'][:80]}")
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
            print(f"{name:22s} B={batch:<5d} baseline {b:9.2f} ms  head {h:9.2f} ms  "
                  f"ratio {h / b:5.3f}  (baseline spread {spread:5.1%})")
        worst = max((r["ratio_head_over_baseline"] for r in rows if r["status"] == "ok"),
                    default=0.0)
        record = {"arms": arm_names, "baseline_sha": a.baseline_sha, "device": a.device,
                  "set": a.which,
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
