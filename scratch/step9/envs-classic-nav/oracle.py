"""Oracle proof for the Workstream-A migration: the same pixels, from two different trees.

WHAT IT PROVES. The Step-8 behavioral oracle (plan §7) says the architecture may change and the
behavior may not. This is ``scratch/step9/foundation/pusht_oracle.py``'s argument, run over all TEN
cells this workstream owns instead of one: the five classic-control cells and the five navigation
cells, each of which has a FROZEN STEP-4 CORPUS at ``examples/results/step4/states/<Cell>.npz``.
Every corpus is pushed through BOTH trees in two separate interpreters — a detached ``git worktree``
of the Step-8 baseline SHA and this branch, the same venv — and the results compared byte for byte:

  1. the frames: ``render(states)`` at the registered default size, SHA-256 over the raw uint8 bytes,
     and again rendered from explicit packed rows;
  2. ``describe()``: the dict a consumer reads to decide whether these frames are usable;
  3. the style layout: ``style_names``, ``style_dim`` and the default (official) appearance;
  4. the registry row: name, tier, decoder, state_dim, reference_id — and ``module`` SEPARATELY, see
     below;
  5. the packed layout itself: ``state_dim``/``state_names``, and ``set_state``/``get_state``
     round-tripping the corpus;
  6. ``sample_states`` at a fixed seed — the env's own reset distribution.

THE ONE EXPECTED DIFFERENCE, declared here rather than discovered in the diff: the five navigation
cells MOVED MODULE. ``swm_nav.py`` and ``simple_point_maze.py`` were merged into ``navigation.py``
(plan §3), so ``registry_table()[i]["module"]`` — and therefore ``spec(name)["family"]`` and
``list_envs(family=...)`` — changed for them. The old import paths still resolve (``sys.modules``
aliases) and this script CHECKS that they do, and checks that no other column of the row moved. It is
recorded under ``expected_differences`` and is not counted as unexplained.

It also checks the claims that are NEW in Step 9 and therefore have no baseline to compare against,
per cell and per device: ``transition`` equals the stateful ``step`` to the bit, ``unroll`` equals
repeated ``transition`` over four steps (the horizon at which a State that had rounded
classic control's float64 dynamics to float32 would start to drift while a one-step check still
passed), and ``render(State) == render(packed rows)``.

Usage::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-classic-nav/oracle.py
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

#: The ten cells, and the frozen corpus each is replayed through. All ten have one.
CELLS = ("CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous",
         "TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "SimplePointMaze")

#: Run inside ONE tree, printing a JSON record on stdout. Written as a string and executed with
#: ``PYTHONPATH`` pointed at the tree under test, because the two trees define the same module names
#: and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, sys
import numpy as np, torch

import stable_envs as se

cells = json.loads(sys.argv[1])
states_dir = sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

table = {r["name"]: r for r in se.registry_table()}
out = {"cells": {}, "import_errors": (se.import_errors() if hasattr(se, "import_errors")
                                      else se.envs.import_errors())}
for name in cells:
    corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)
    n = int(corpus.shape[0])
    cls = se.REGISTRY[name]
    try:
        env = cls(num_envs=n, device=dev)
    except TypeError:
        env = cls(batch=n, device=dev)
    env.set_state(corpus)
    rows = env.get_state().detach().cpu().numpy()
    frames = env.render().detach().cpu().numpy()
    frames_from_rows = env.render(torch.as_tensor(corpus)).detach().cpu().numpy()
    out["cells"][name] = {
        "device": dev,
        "state_dim": int(env.state_dim),
        "state_names": list(env.state_names),
        "size": int(env.size),
        "corpus_states": n,
        "roundtrip_exact": bool(np.array_equal(rows, corpus)),
        "rows_sha256": h(rows),
        "frames_shape": list(frames.shape),
        "frames_sha256": h(frames),
        "frames_from_rows_sha256": h(frames_from_rows),
        "frames_mean": float(frames.mean()),
        "describe": json.loads(json.dumps(env.describe(), default=str)),
        "style_names": list(env.style_names),
        "style_dim": int(env.style_dim),
        "default_style_sha256": h(np.asarray(env.default_style(), np.float32)),
        "registry_row": table[name],
        "sample_states_sha256": h(np.asarray(env.sample_states(16, np.random.default_rng(0)),
                                             np.float32)),
        "variation_names": list(env.variation_space.names()),
        "parity_interior_min": env.parity_interior_min,
        "parity_all_exact_min": env.parity_all_exact_min,
        "fidelity_tier": env.fidelity_tier,
        "render_decoder": env.render_decoder,
    }
    if hasattr(env, "close"):
        env.close()
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the migrated tree only (there is nothing to compare them against).
FUNCTIONAL = r'''
import json, sys
import numpy as np, torch
import stable_envs as snv

cells = json.loads(sys.argv[1])
states_dir = sys.argv[2]
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    for name in cells:
        corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)[:16]
        n = int(corpus.shape[0])
        env = getattr(snv, name)(num_envs=n, device=dev)
        keys = [tuple(k) for k, _, _ in env._layout()]
        g = torch.Generator().manual_seed(0)
        lo, hi = float(env.action_low), float(env.action_high)
        def acts(steps=None):
            shape = (n, env.action_dim) if steps is None else (steps, n, env.action_dim)
            return (lo + (hi - lo) * torch.rand(*shape, generator=g)).to(env.device)

        env.set_state(corpus)
        held = env.get_state().clone()
        a = acts()
        functional = env.transition(env.state, a)
        pure = bool(torch.equal(env.get_state(), held))
        stateful = env.step(a)["state"]
        same = bool(torch.equal(env.pack(functional), env.pack(stateful))) and all(
            bool(torch.equal(functional[k], stateful[k])) for k in keys)

        env.set_state(corpus)
        A = acts(4)
        traj = env.unroll(env.state, A)
        cur = env.state
        unroll_ok = True
        for t in range(4):
            cur = env.transition(cur, A[t])
            unroll_ok &= all(bool(torch.equal(traj[t][k], cur[k])) for k in keys)
        env.set_state(corpus)
        for t in range(4):
            env.step(A[t])
        chain_ok = all(bool(torch.equal(traj[3][k], env.state[k])) for k in keys)

        env.set_state(corpus)
        st = env.state
        render_ok = bool(torch.equal(env.render(st), env.render(env.pack(st))))
        graphed = bool(getattr(env, "_step_graph", None) is not None)
        out.append({"device": dev, "cell": name,
                    "transition_is_pure": pure,
                    "transition_equals_step_bitwise": same,
                    "unroll_equals_repeated_transition": unroll_ok,
                    "unroll_equals_four_stateful_steps": chain_ok,
                    "stateful_path_was_a_cuda_graph": graphed,
                    "render_state_equals_render_rows": render_ok})
        env.close()

# the deprecated navigation import paths still resolve to the merged module
from stable_envs.envs import navigation, simple_point_maze, swm_nav
shims = {"swm_nav": swm_nav is navigation, "simple_point_maze": simple_point_maze is navigation}
print("@@JSON@@" + json.dumps({"checks": out, "old_module_paths_resolve": shims}))
'''


def _run(code: str, tree: Path, python: str, *args: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", code, *args], capture_output=True, text=True,
                       cwd=str(tree), env=env)
    if r.returncode != 0:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")][-1]
    return json.loads(line[len("@@JSON@@"):])


def _baseline_worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``. Returns the path and whether this call created it."""
    existing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout
    for block in existing.split("\n\n"):
        if sha[:12] in block and "step9-oracle-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-oracle-baseline-{sha[:12]}"
    if not path.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                       check=True, capture_output=True)
        return path, True
    return path, False


#: Every key the two trees must agree on, and the sentence that says what a difference would mean.
COMPARED = {
    "device": "the device the comparison arm ran on",
    "state_dim": "the packed row's width",
    "state_names": "the packed row's column names",
    "size": "the registered default frame size",
    "corpus_states": "how many frozen states were replayed",
    "roundtrip_exact": "set_state/get_state round-tripping the frozen corpus",
    "rows_sha256": "the packed rows the renderer was handed",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "the official appearance",
    "registry_row": "the registry row (module compared separately: see the module docstring)",
    "sample_states_sha256": "the env's own reset distribution at a fixed seed",
    "variation_names": "the variation taxonomy, leaf for leaf",
    "parity_interior_min": "the declared interior-parity floor",
    "parity_all_exact_min": "the declared all-pixel floor",
    "fidelity_tier": "the fidelity tier",
    "render_decoder": "the decoder declaration",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=",".join(CELLS))
    #: WHICH DEVICE THE COMPARISON ARM RUNS ON. ``cpu`` is the default and is the whole story for a
    #: local run; on a CUDA host ``--device cuda`` is the ONE way to compare the fused render paths
    #: (``render/gfxdraw_classic`` and ``render/navdot`` are CUDA-only, so a cpu comparison never
    #: executes a line of them) between the two trees.
    ap.add_argument("--device", default="cpu")
    #: The tree is an rsync copy inside a clone on a cluster, so ``git rev-parse HEAD`` there names
    #: the clone's HEAD and not the branch that was measured. Passed explicitly rather than guessed.
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-classic-nav"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        before = _run(PROBE, baseline, a.python, json.dumps(cells),
                      str(baseline / "examples/results/step4/states"), a.device)
        after = _run(PROBE, ROOT, a.python, json.dumps(cells),
                     str(ROOT / "examples/results/step4/states"), a.device)
        fn = _run(FUNCTIONAL, ROOT, a.python, json.dumps(cells),
                  str(ROOT / "examples/results/step4/states"))

        diffs: dict = {}
        expected: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                bv, mv = b[k], m[k]
                if k == "registry_row":
                    # the module column is the declared, coordinator-mandated file move
                    if bv.get("module") != mv.get("module"):
                        expected.setdefault(name, {})["registry_row.module"] = {
                            "baseline": bv.get("module"), "migrated": mv.get("module"),
                            "why": "plan §3 merged swm_nav.py and simple_point_maze.py into "
                                   "navigation.py; the old import paths still resolve (checked)"}
                    bv = {kk: vv for kk, vv in bv.items() if kk != "module"}
                    mv = {kk: vv for kk, vv in mv.items() if kk != "module"}
                if bv != mv:
                    diffs.setdefault(name, {})[k] = {"baseline": bv, "migrated": mv, "means": why}

        checks = fn["checks"]
        functional_ok = all(c["transition_is_pure"] and c["transition_equals_step_bitwise"]
                            and c["unroll_equals_repeated_transition"]
                            and c["unroll_equals_four_stateful_steps"]
                            and c["render_state_equals_render_rows"] for c in checks)
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "compare_device": a.device,
            "workstream": "step9/envs-classic-nav",
            "cells": cells,
            "corpus": "examples/results/step4/states/<Cell>.npz",
            "python": a.python,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "expected_differences": expected,
            "identical": not diffs,
            "old_module_paths_resolve": fn["old_module_paths_resolve"],
            "functional_checks": checks,
            "functional_ok": functional_ok,
            "baseline": before,
            "migrated": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / a.name).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

        print(json.dumps({k: record[k] for k in ("baseline_sha", "head_sha", "identical",
                                                 "unexplained_differences", "expected_differences",
                                                 "old_module_paths_resolve", "functional_ok")},
                         indent=1))
        for name in cells:
            print(f"  {name:24s} frames {record['migrated']['cells'][name]['frames_sha256'][:16]}"
                  f"  states {record['migrated']['cells'][name]['corpus_states']:4d}")
        ok = (record["identical"] and functional_ok
              and all(fn["old_module_paths_resolve"].values()))
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
