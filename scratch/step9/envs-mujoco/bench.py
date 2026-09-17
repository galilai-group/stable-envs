"""Paired, alternating before/after CPU timings for the Workstream-F families (plan §8, §21).

WHY ALTERNATING. A "before" block followed by an "after" block measures the machine's mood as much as
the code: a thermal ramp, another Step-9 agent starting its suite on this laptop, or a frequency step
lands entirely on one leg. So the legs are interleaved -- baseline, head, baseline, head, ... -- in
separate interpreters, and the statistic is the median over rounds of each leg. Same venv, same host,
same process shape, same states; the only difference is which tree is on ``PYTHONPATH``.

WHAT IS TIMED, and why there are TWO hot paths here rather than one. This workstream's patch touches
two caller-visible paths and they fail differently:

``render``  every family in this workstream is first of all a renderer, and the branch now carries
            Workstream G's 2-D backend consolidation underneath nine of them (tilemap, raster2d,
            scene, gfxdraw, navdot, prng). The frames are byte-identical (see ``oracle.py``); what
            this set asks is whether they cost the same. One cell per distinct backend shape rather
            than all twenty: the MuJoCo CPU renderer (Reacher/Pusher), ``raster2d`` at 600 px and at
            its dispatch-bound primitive count (Intersection, Roundabout), the non-square frame
            (Parking, 300x600), the ``tilemap`` blit (Crafter at 64 px, MiniGridEmpty, Sokoban), the
            baked-plate path (MiniHack at 336x1264) and the POV gather (MiniGridPOVFetch).

``step``    the seven steppable cells are the ONLY place this patch changed executable code rather
            than declarations: ``_advance`` became ``_advance_into(buffers...)`` in minigrid.py and
            sokoban.py so that ``transition`` is pure by construction, and ``step`` now returns a
            ``StepResult`` TensorDict instead of a bare tensor. Ruling 7 accepted a StepResult cost
            of +12-28 µs on trivially cheap steps; these steps are not trivially cheap, but the
            indirection through a caller-named buffer is exactly the kind of change that can cost a
            copy per step, so it is measured and not assumed. The baseline tree has ``step`` on these
            cells too (returning a tensor), so the pairing is direct.

The bar (plan §8): a repeatable regression of roughly >10 % on a hot path blocks merge. ``ratio`` is
head/baseline wall time, so >1.10 is a regression, and the per-leg spread printed beside it says
whether this machine could resolve it at all. ``--arms head,head`` times ONE tree against itself in
the same alternating shape, which is how a ratio that is a property of the laptop is told apart from
one that is a property of the patch; run it beside any ratio worth believing.

Usage::

    export SE9_EXTRA_PYTHONPATH=$(scratch/step9/envs-mujoco/oracle.py --print-extra-pythonpath)
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-mujoco/bench.py --set render
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-mujoco/bench.py --set step
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-mujoco/bench.py --set maze2d \
        --python /Users/samib/se8-fetch-venv/bin/python

GPU timings are NOT this script's job: ruling 8 puts every GPU number in the one post-Wave-1 job, and
``scratch/step9/envs-mujoco/gpu_oracle.sbatch`` carries the section that produces them.
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

#: ``(config, batch)`` at the registered frame size. Batches are capped so a 600 px or 336x1264 CPU
#: raster finishes in a bounded time: a ratio is a ratio at any batch, and the batch is recorded.
SETS: dict[str, list[tuple[str, int]]] = {
    "render": [
        ("MujocoReacher", 8),          # MuJoCo CPU renderer, 480 px
        ("MujocoPusher", 8),           # same path, heavier scene
        ("Highway", 16),               # raster2d, 600x150
        ("Roundabout", 8),             # raster2d, 600 px, the dispatch-bound primitive count
        ("Intersection", 8),           # raster2d, 600 px
        ("Parking", 16),               # raster2d, 300x600 -- the non-square frame frame_shape fixes
        ("Crafter", 32),               # tilemap blit, 64 px
        ("MiniHack", 8),               # baked plates, 336x1264
        ("MiniGridEmpty", 32),         # tilemap, top-down
        ("MiniGridPOVFetch", 32),      # tilemap, POV gather
        ("Sokoban", 32),               # tilemap, 160 px, 16x16 art
    ],
    "step": [
        ("MiniGridEmpty", 64),
        ("MiniGridDoorKey", 64),
        ("MiniGridLavaCrossing", 64),
        ("BabyAIGoToLocal", 64),
        ("Sokoban", 64),
    ],
    #: gymnasium-robotics, so a different interpreter: pass --python /Users/samib/se8-fetch-venv/...
    "maze2d": [
        ("PointMazeOpen", 8),
        ("PointMazeUMaze", 8),
        ("PointMazeMedium", 8),
        ("PointMazeLarge", 8),
    ],
}

#: Sample each cell's states ONCE, in the BASELINE tree, and write them to a temp ``.npz`` the legs
#: load. Two reasons, and the second is the important one:
#:
#:   * ``sample_states`` is SETUP, not the timed region, and for MiniGrid and Sokoban it is the
#:     dominant cost of a leg -- Sokoban rejection-samples whole solvable rooms. Re-drawing it in
#:     every one of the 100+ legs multiplied the run by ~20x and measured nothing;
#:   * both legs then time the SAME states. Otherwise a ratio mixes the patch with whichever poses
#:     each leg happened to draw, and for a renderer whose cost depends on what is on screen (how
#:     many vehicles, how much lava) that is a real confound and not a rounding one.
#:
#: This is ``oracle.py``'s argument for its own corpus, applied to timings, and the same shape as
#: ``scratch/step9/render-2d/bench.py``'s frozen ``CORPUS`` -- except that no cell here HAS a frozen
#: corpus (see oracle.py's docstring), so it is sampled at a fixed seed instead.
SAMPLE = r'''
import json, sys
import numpy as np
import stable_envs as se

cells, out_dir = json.loads(sys.argv[1]), sys.argv[2]
for name, batch in cells:
    env = se.REGISTRY[name](batch=int(batch), device="cpu")
    s = np.asarray(env.sample_states(int(batch), np.random.default_rng(0)), np.float32)
    np.savez(f"{out_dir}/{name}.npz", states=s)
    getattr(env, "close", lambda: None)()      # the Step-8 baseline tree has no Env.close
print("@@JSON@@" + json.dumps({"ok": True}))
'''

#: Run inside ONE tree and time ONE cell. Written as a string and executed with ``PYTHONPATH``
#: pointed at the tree under test, because the two trees define the same module names and cannot both
#: be imported into one interpreter. ``batch=`` and not ``num_envs=``: it is the spelling BOTH trees
#: accept, and which spelling is accepted is the ratchet's question, not this script's.
LEG = r'''
import json, sys, time
import numpy as np, torch
import stable_envs as se

name, batch, device = sys.argv[1], int(sys.argv[2]), sys.argv[3]
warmup, reps, what = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
states_dir = sys.argv[7]

env = se.REGISTRY[name](batch=batch, device=device)
s = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)
env.set_state(s)
sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)

if what == "render":
    call = env.render
else:
    # (B, 1) float actions, the spelling both trees take. Fixed, not random: a ratio must not depend
    # on which branch of the transition each leg happened to draw.
    act = torch.full((batch, env.action_dim), 2.0, dtype=torch.float32, device=env.device)
    call = lambda: env.step(act)

for _ in range(warmup):
    call(); sync()
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); call(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
print("@@JSON@@" + json.dumps({"ms": ts, "size": int(env.size)}))
'''


def _worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``, reusing ``oracle.py``'s if it is still checked out."""
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-envs-mujoco-oracle-baseline" in block or "step9-envs-mujoco-bench" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-envs-mujoco-bench-{sha[:12]}"
    if path.exists():
        return path, False
    subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                   check=True, capture_output=True)
    return path, True


def _leg_sample(tree: Path, python: str, cells: list[tuple[str, int]], states_dir: str):
    """Draw every cell's states ONCE, in the baseline tree, so both legs time the same input."""
    env = dict(os.environ)
    extra = os.environ.get("SE9_EXTRA_PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tree}:{extra}" if extra else str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", SAMPLE, json.dumps(cells), states_dir],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        return None, f"{r.stdout[-800:]}\n{r.stderr[-800:]}"
    return json.loads(hit[-1][len("@@JSON@@"):]), None


def _leg(tree: Path, python: str, name: str, batch: int, device: str, warmup: int, reps: int,
         what: str, states_dir: str):
    env = dict(os.environ)
    extra = os.environ.get("SE9_EXTRA_PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tree}:{extra}" if extra else str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    r = subprocess.run([python, "-c", LEG, name, str(batch), device, str(warmup), str(reps), what,
                        states_dir], capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        return None, f"{r.stdout[-600:]}\n{r.stderr[-600:]}"
    return json.loads(hit[-1][len("@@JSON@@"):]), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", dest="which", default="render", choices=sorted(SETS))
    ap.add_argument("--rounds", type=int, default=5, help="alternating before/after rounds")
    ap.add_argument("--reps", type=int, default=5, help="timed calls per leg")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-mujoco"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--arms", default="baseline,head",
                    help="which TREE each leg runs. ``head,head`` times the SAME tree against "
                         "itself in the same alternating shape, giving the floor under which a "
                         "ratio means nothing. It is not expected to be exactly 1.000.")
    a = ap.parse_args()
    arm_names = a.arms.split(",")
    if len(arm_names) != 2 or any(x not in ("baseline", "head") for x in arm_names):
        raise SystemExit("--arms takes exactly two of baseline|head")
    what = "render" if a.which in ("render", "maze2d") else "step"

    baseline, created = _worktree(a.baseline_sha)
    states_tmp = tempfile.TemporaryDirectory()
    states_dir = states_tmp.name
    try:
        got, e = _leg_sample(baseline, a.python, SETS[a.which], states_dir)
        if got is None:
            raise SystemExit(f"could not sample states in the baseline tree:\n{e}")
        rows = []
        for name, batch in SETS[a.which]:
            legs: dict[str, list[float]] = {"baseline": [], "head": []}
            size = None
            err = None
            for _ in range(a.rounds):
                trees = {"baseline": baseline, "head": ROOT}
                for label, arm in (("baseline", arm_names[0]), ("head", arm_names[1])):
                    got, e = _leg(trees[arm], a.python, name, batch, a.device, a.warmup, a.reps,
                                  what, states_dir)
                    if got is None:
                        err = e
                        break
                    legs[label].append(statistics.median(got["ms"]))
                    size = got["size"]
                if err:
                    break
            if err:
                rows.append({"config": name, "batch": batch, "status": "skipped",
                             "reason": err.strip().splitlines()[-1][:160] if err.strip() else "?"})
                print(f"{name:24s} B={batch:<5d} skipped: {rows[-1]['reason'][:80]}")
                continue
            b = statistics.median(legs["baseline"])
            h = statistics.median(legs["head"])
            spread = (max(legs["baseline"]) - min(legs["baseline"])) / b if b else 0.0
            rows.append({"config": name, "batch": batch, "size": size, "path": what,
                         "status": "ok",
                         "baseline_ms": round(b, 3), "head_ms": round(h, 3),
                         "ratio_head_over_baseline": round(h / b, 4),
                         "baseline_leg_spread": round(spread, 4),
                         "baseline_rounds_ms": [round(x, 3) for x in legs["baseline"]],
                         "head_rounds_ms": [round(x, 3) for x in legs["head"]]})
            print(f"{name:24s} B={batch:<5d} baseline {b:9.3f} ms  head {h:9.3f} ms  "
                  f"ratio {h / b:5.3f}  (baseline spread {spread:5.1%})")
        worst = max((r["ratio_head_over_baseline"] for r in rows if r["status"] == "ok"),
                    default=0.0)
        record = {"arms": arm_names, "baseline_sha": a.baseline_sha, "device": a.device,
                  "set": a.which, "path": what,
                  "host": os.uname().nodename, "python": a.python, "rounds": a.rounds,
                  "reps_per_leg": a.reps, "rows": rows, "worst_ratio": worst,
                  "states": "sampled ONCE in the BASELINE tree with sample_states(batch, "
                            "default_rng(0)); both legs time the same states, and sampling is not "
                            "inside the timed region",
                  "regression_bar": 1.10, "blocks_merge": bool(worst > 1.10),
                  "extra_pythonpath": (os.environ.get("SE9_EXTRA_PYTHONPATH", "").split(":")
                                       if os.environ.get("SE9_EXTRA_PYTHONPATH") else []),
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
        states_tmp.cleanup()
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
