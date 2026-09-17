"""Oracle proof for the Step-9 render-3d workstream: the same pixels, from two different trees.

WHAT IT PROVES. Plan §7 and §27 say the architecture may change and the behaviour may not. For
``render/raycast3d.py`` and ``render/cpu3d.py`` "the behaviour" is the frames, so this runs EVERY
consumer cell that reaches either module through the PUBLIC path -- ``cls(batch=n,
device=...)``, ``set_state``, ``render()`` -- in two separate interpreters, one with the Step-8
baseline tree on ``PYTHONPATH`` and one with this branch, and compares SHA-256 over the raw uint8
frame bytes. Nothing is compared against a stored digest: both sides are computed here, now, in the
same venv on the same host, so a difference can only be ours.

WHICH CELLS, AND HOW THEY WERE CHOSEN. Not by reading the imports: by instrumenting them. Every
registered environment was constructed on CPU and rendered with ``raycast3d.cast_nearest``,
``cast_nearest_torch``, ``any_blocking``, ``static_depth``, ``composite`` and ``cpu3d.render``
wrapped in counters, and the cells below are the ones that actually called something (the same
probe is re-run inside each arm and the call set is itself compared, so a cell that silently stops
using the backend is a difference rather than a pass). ``backends`` in the record is that call set.

WHICH STATES. The frozen corpora wherever one exists -- ``examples/results/step4/states/*.npz`` for
the DM Control parity and trackcom cells and the OGBench mazes, ``examples/results/step7/**`` for
Manipulator and Quadruped -- and, for the four gym-MuJoCo cells and DMCPointMassEasy, which never
had a frozen corpus, the env's own ``sample_states`` at a fixed seed. The digest of the states
themselves is recorded and compared too, so "the same frames" is never an artefact of the two arms
having been handed different inputs.

THE SLOW PATHS ARE CHECKED AS WELL. ``PRIMITIVE_BOUNDS``, ``COVER_COMPACTION`` and
``COMPOSITE_COMPACT_MERGE`` each select between a fast expression and the readable one it must be
bit-identical to. A refactor that preserved only the fast path would pass a naive frame comparison,
so two cells are additionally rendered with each switch off, in both trees.

Usage (CPU arm, locally)::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/render-3d/oracle.py

GPU arm (inside the SLURM job, see ``gpu_oracle.sbatch``)::

    python scratch/step9/render-3d/oracle.py --devices cpu,cuda
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: ``(config, corpus)``; ``None`` means the cell has no frozen corpus and its own ``sample_states``
#: at seed 0 is used instead. Every one of these was measured to reach ``raycast3d`` or ``cpu3d``.
CELLS: list[tuple[str, str | None]] = [
    # DM Control parity cells: raycast3d's analytic cast over a baked static plate
    ("DMCAcrobotSwingup", "examples/results/step4/states/DMCAcrobotSwingup.npz"),
    ("DMCBallInCupCatch", "examples/results/step4/states/DMCBallInCupCatch.npz"),
    ("DMCCartpoleBalance", "examples/results/step4/states/DMCCartpoleBalance.npz"),
    ("DMCCartpoleSwingup", "examples/results/step4/states/DMCCartpoleSwingup.npz"),
    ("DMCFingerSpin", "examples/results/step4/states/DMCFingerSpin.npz"),
    ("DMCFingerTurnEasy", "examples/results/step4/states/DMCFingerTurnEasy.npz"),
    ("DMCFingerTurnHard", "examples/results/step4/states/DMCFingerTurnHard.npz"),
    ("DMCPendulumSwingup", "examples/results/step4/states/DMCPendulumSwingup.npz"),
    ("DMCReacherEasy", "examples/results/step4/states/DMCReacherEasy.npz"),
    ("DMCReacherHard", "examples/results/step4/states/DMCReacherHard.npz"),
    ("DMCPointMassEasy", None),
    # Step-7 raycast3d cells
    ("DMCManipulatorBringBall",
     "examples/results/step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz"),
    # the trackcom cells: cpu3d's own shading law, raycast3d's intersectors for the movers
    ("DMCCheetahRun", "examples/results/step4/states/DMCCheetahRun.npz"),
    ("DMCHopperHop", "examples/results/step4/states/DMCHopperHop.npz"),
    ("DMCHumanoidStand", "examples/results/step4/states/DMCHumanoidStand.npz"),
    ("DMCWalkerWalk", "examples/results/step4/states/DMCWalkerWalk.npz"),
    ("DMCQuadrupedWalk",
     "examples/results/step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz"),
    # gym MuJoCo wrappers, raycast3d over a baked plate; no frozen corpus, so seeded sampling
    ("MujocoInvertedPendulum", None),
    ("MujocoInvertedDoublePendulum", None),
    ("MujocoPusher", None),
    ("MujocoReacher", None),
    # OGBench mazes: raycast3d's composite. Need the ``ogbench`` package, so these are the Oscar arm.
    ("PointmazeMediumNavigate", "examples/results/step4/states/PointmazeMediumNavigate.npz"),
    ("PointmazeLargeNavigate", "examples/results/step4/states/PointmazeLargeNavigate.npz"),
    ("PointmazeGiantNavigate", "examples/results/step4/states/PointmazeGiantNavigate.npz"),
    ("PointmazeTeleportNavigate", "examples/results/step4/states/PointmazeTeleportNavigate.npz"),
    ("AntmazeMediumExplore", "examples/results/step4/states/AntmazeMediumExplore.npz"),
    ("AntmazeLargeExplore", "examples/results/step4/states/AntmazeLargeExplore.npz"),
    ("AntmazeGiantNavigate", "examples/results/step4/states/AntmazeGiantNavigate.npz"),
    ("AntmazeTeleportExplore", "examples/results/step4/states/AntmazeTeleportExplore.npz"),
    ("HumanoidmazeMediumNavigate", "examples/results/step4/states/HumanoidmazeMediumNavigate.npz"),
    ("HumanoidmazeLargeNavigate", "examples/results/step4/states/HumanoidmazeLargeNavigate.npz"),
    ("HumanoidmazeGiantNavigate", "examples/results/step4/states/HumanoidmazeGiantNavigate.npz"),
    # maze2d (gymnasium-robotics) and the OGB manipulation cells, both raycast3d consumers
    ("PointMazeUMaze", None),
    ("PointMazeOpen", None),
    ("PointMazeMedium", None),
    ("PointMazeLarge", None),
    ("OGBCube", "examples/results/step7/ogb-manipulation/states/OGBCube.npz"),
    ("OGBScene", "examples/results/step7/ogb-manipulation/states/OGBScene.npz"),
]

#: Cells rendered a second and third time with the fast-path switches off, so the refactor is shown
#: to have preserved the READABLE expression each switch selects and not only the shipped one.
SWITCH_CELLS = ("DMCReacherHard", "DMCBallInCupCatch")

#: Run inside ONE tree, printing a JSON record on stdout. A string rather than an import because the
#: two trees define the same module names and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, sys, traceback
import numpy as np, torch

import stable_envs as se
from stable_envs.render import raycast3d as R3, cpu3d as C3

cells = json.loads(sys.argv[1])
devices = sys.argv[2].split(",")
switch_cells = set(json.loads(sys.argv[3]))
nstates = int(sys.argv[4])

HITS = set()
for mod, name in ((R3, "cast_nearest"), (R3, "cast_nearest_torch"), (R3, "any_blocking"),
                  (R3, "any_blocking_torch"), (R3, "static_depth"), (R3, "composite"),
                  (R3, "_composite_window"), (C3, "render")):
    orig = getattr(mod, name)
    def wrap(orig=orig, tag="%s.%s" % (mod.__name__.rsplit(".", 1)[-1], name)):
        def f(*a, **k):
            HITS.add(tag)
            return orig(*a, **k)
        return f
    setattr(mod, name, wrap())


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def states_for(env, corpus):
    if corpus is None:
        return np.asarray(env.sample_states(nstates, np.random.default_rng(0)), np.float32)
    return np.load(corpus)["states"].astype(np.float32)[:nstates]


def frames(env, s):
    env.set_state(s)
    return env.render().detach().cpu().numpy()


out = {}
for name, corpus in cells:
    rec = {}
    out[name] = rec
    for dev in devices:
        key = "device_%s" % dev
        if dev == "cuda" and not torch.cuda.is_available():
            rec[key] = {"status": "skipped", "reason": "no cuda device"}
            continue
        HITS.clear()
        try:
            cls = se.REGISTRY[name]
            env = cls(batch=1, device=dev)          # cheap probe of support before the real run
            env.close() if hasattr(env, "close") else None
        except Exception as e:
            rec[key] = {"status": "skipped", "reason": "%s: %s" % (type(e).__name__, str(e)[:160])}
            continue
        try:
            s = None
            cls = se.REGISTRY[name]
            env = cls(batch=nstates, device=dev)
            s = states_for(env, corpus)
            env = cls(batch=len(s), device=dev) if len(s) != nstates else env
            f = frames(env, s)
            r = {"status": "ok", "states_sha256": h(s), "n_states": int(len(s)),
                 "size": int(env.size), "frames_shape": list(f.shape), "frames_sha256": h(f),
                 "frames_mean": round(float(f.mean()), 9),
                 "backends": sorted(HITS)}
            # the frames must not depend on how the rows arrive
            r["frames_from_rows_sha256"] = h(env.render(torch.as_tensor(s, device=env.device))
                                             .detach().cpu().numpy())
            if name in switch_cells:
                sw = {}
                for label, attrs in (("primitive_bounds_off", {"PRIMITIVE_BOUNDS": False}),
                                     ("cover_compaction_off", {"COVER_COMPACTION": False}),
                                     ("composite_window", {"COMPOSITE_COMPACT_MERGE": False})):
                    saved = {k: getattr(R3, k) for k in attrs}
                    for k, v in attrs.items():
                        setattr(R3, k, v)
                    try:
                        sw[label] = h(frames(env, s))
                    finally:
                        for k, v in saved.items():
                            setattr(R3, k, v)
                r["switches"] = sw
            rec[key] = r
            try:
                env.close()
            except Exception:
                pass
        except Exception as e:
            rec[key] = {"status": "error", "reason": "%s: %s" % (type(e).__name__, str(e)[:200]),
                        "traceback": traceback.format_exc()[-800:]}
print("@@JSON@@" + json.dumps(out))
'''


def _run_probe(tree: Path, python: str, devices: str, nstates: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    cells = json.dumps([[n, (str(tree / c) if c else None)] for n, c in CELLS])
    r = subprocess.run([python, "-c", PROBE, cells, devices, json.dumps(list(SWITCH_CELLS)),
                        str(nstates)], capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    return json.loads(hit[-1][len("@@JSON@@"):])


def _baseline_worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``, so the comparison is against the tree and not a memory of it."""
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


#: The keys that must agree, and what a difference in each one would mean.
COMPARED = {
    "status": "whether the cell renders at all on this device",
    "reason": "why a cell was skipped, when it was",
    "states_sha256": "the states the two arms were handed",
    "n_states": "how many states were rendered",
    "size": "the registered default frame size",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "backends": "which of the two backends' entry points the cell actually called",
    "switches": "the readable expression behind each fast-path switch",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--devices", default="cpu", help="comma separated: cpu, cuda")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-3d"))
    ap.add_argument("--tag", default=None, help="suffix for the record file (e.g. the host)")
    ap.add_argument("--arms", default="baseline,head",
                    help="which TREE each arm runs: baseline|head for each. The default compares the "
                         "two trees. ``--arms baseline,baseline`` runs the SAME tree twice in the "
                         "same 38-cell process shape, which is how a cell that is not reproducible "
                         "across processes is told apart from a cell this branch changed.")
    a = ap.parse_args()

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        arms = a.arms.split(",")
        if len(arms) != 2 or any(x not in ("baseline", "head") for x in arms):
            raise SystemExit("--arms takes exactly two of baseline|head, e.g. baseline,baseline")
        trees = {"baseline": baseline, "head": ROOT}
        before = _run_probe(trees[arms[0]], a.python, a.devices, a.n_states)
        after = _run_probe(trees[arms[1]], a.python, a.devices, a.n_states)

        diffs, cells = {}, {}
        for name, _ in CELLS:
            for dev in a.devices.split(","):
                key = f"device_{dev}"
                b, m = before[name].get(key, {}), after[name].get(key, {})
                d = {k: {"baseline": b.get(k), "head": m.get(k), "means": why}
                     for k, why in COMPARED.items() if b.get(k) != m.get(k)}
                if d:
                    diffs[f"{name}/{dev}"] = d
                cells[f"{name}/{dev}"] = m.get("status", "missing")
        record = {
            "arms": arms,
            "baseline_sha": a.baseline_sha,
            "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       capture_output=True, text=True).stdout.strip(),
            "python": a.python,
            "devices": a.devices.split(","),
            "n_states_requested": a.n_states,
            "host": os.uname().nodename,
            "cells": cells,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "identical": not diffs,
            "baseline": before,
            "head": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"oracle{'_' + a.tag if a.tag else ''}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        ran = sum(1 for v in cells.values() if v == "ok")
        print(json.dumps({"arms": arms, "baseline_sha": record["baseline_sha"],
                          "head_sha": record["head_sha"],
                          "identical": record["identical"], "cells_ok": ran,
                          "cells": cells, "unexplained_differences": diffs}, indent=1))
        print(f"wrote {out / fn}")
        print("ORACLE:", "PASS" if record["identical"] and ran else "FAIL")
        return 0 if (record["identical"] and ran) else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
