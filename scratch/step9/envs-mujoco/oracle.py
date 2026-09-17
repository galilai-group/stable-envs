"""Oracle proof for the Workstream-F migration: the same pixels, from two different trees.

WHAT IT PROVES. The Step-8 behavioral oracle (plan §7) says the architecture may change and the
behavior may not. This is ``scratch/step9/foundation/pusht_oracle.py``'s argument run over the cells
this workstream owns — the MuJoCo/nvdiff wrappers and the other non-42-id families — through a
detached ``git worktree`` of the Step-8 baseline SHA and through this branch, in ONE venv, comparing
byte for byte:

  1. the frames: ``render(states)`` at the registered default size, SHA-256 over the raw uint8
     bytes, and again rendered from explicit packed rows;
  2. ``describe()``: the dict a consumer reads to decide whether these frames are usable;
  3. the style layout: ``style_names``, ``style_dim`` and the default (official) appearance;
  4. the registry row: name, tier, decoder, state_dim, reference_id — and ``module`` separately;
  5. the packed layout itself: ``state_dim``/``state_names``, and ``set_state``/``get_state``
     round-tripping the states;
  6. ``sample_states`` at a fixed seed — the env's own reset distribution;
  7. ``variation_space.names()`` — the taxonomy, leaf for leaf — and the fidelity declarations.

WHERE THE STATES COME FROM, and why this differs from Workstream A's script. **Not one cell this
workstream owns has a frozen Step-4 corpus**: ``examples/results/step4/states/`` holds 49 corpora and
every one of them belongs to another workstream (PushT, classic control, navigation, DM Control,
Fetch, OGBench, Craftax). There is therefore nothing frozen to replay here, and inventing a corpus
would freeze something Step 4 deliberately did not.

What is used instead is STRICTLY STRONGER than sampling in each tree independently: the BASELINE tree
samples ``sample_states(n, default_rng(0))`` for every cell and writes them to a temp ``.npz``, and
BOTH arms then render THAT array. So the frames are compared over identical inputs, exactly as a
frozen corpus would be, and the reset distribution itself is compared separately (item 6) by having
each arm sample its own and hashing the result — a difference there would show up as a difference in
``sample_states_sha256`` rather than silently changing what was rendered.

It also checks the claims that are NEW in Step 9 and have no baseline to compare against, per cell:
``state`` keys cover the packed row and ``pack(state) == get_state()``; ``render(State) ==
render(rows) == render()``; a cell with NO batched simulator raises the contract's
``NotImplementedError`` from ``step`` and from ``transition`` (and ``has_dynamics`` is False); a cell
WITH one has ``transition`` pure and bit-identical to ``step``, and ``unroll`` equal both to repeated
``transition`` and to repeated ``step``.

Usage::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-mujoco/oracle.py
    MUJOCO_GL=glfw scratch/step9/envs-mujoco/oracle.py --python /Users/samib/se8-fetch-venv/bin/python \
        --cells PointMazeOpen,PointMazeUMaze,PointMazeMedium,PointMazeLarge --name oracle_maze2d.json
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

#: The cells this workstream owns that construct and render on CPU in ``se6-venv`` (with ``minigrid``
#: and ``gym-sokoban`` on ``PYTHONPATH``). The four ``maze2d`` cells need ``gymnasium-robotics`` and
#: are run separately in ``se8-fetch-venv``; the rest of the families own no locally constructible
#: cell on this host and are owed to the post-Wave-1 GPU oracle — see findings §6.
CELLS = ("MujocoInvertedPendulum", "MujocoInvertedDoublePendulum", "MujocoReacher", "MujocoPusher",
         "Highway", "Merge", "Roundabout", "Racetrack", "Parking", "Intersection",
         "Crafter", "MiniHack",
         "MiniGridEmpty", "MiniGridDoorKey", "MiniGridFourRooms", "MiniGridLavaCrossing",
         "MiniGridKeyCorridor", "BabyAIGoToLocal", "MiniGridPOVFetch", "Sokoban")

#: How many states per cell. Small on purpose: ``Intersection`` renders at 600x600 and these are CPU
#: ray-casts and tile composites, not a frozen-corpus replay with a fixed size.
N_STATES = 16

#: Sample the states ONCE, in the baseline tree, so both arms render the same input.
SAMPLE = r'''
import json, sys
import numpy as np
import stable_envs as se

cells, out_dir, n = json.loads(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for name in cells:
    cls = se.REGISTRY[name]
    try:
        env = cls(num_envs=1, device="cpu")
    except TypeError:
        env = cls(batch=1, device="cpu")
    s = np.asarray(env.sample_states(n, np.random.default_rng(0)), np.float32)
    np.savez(f"{out_dir}/{name}.npz", states=s)
    getattr(env, "close", lambda: None)()          # the Step-8 baseline tree has no Env.close
print("@@JSON@@" + json.dumps({"ok": True}))
'''

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
out = {"cells": {}}
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
        "states": n,
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
    getattr(env, "close", lambda: None)()          # the Step-8 baseline tree has no Env.close
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the migrated tree only (there is nothing to compare them against).
FUNCTIONAL = r'''
import json, sys
import numpy as np, torch
import stable_envs as snv
from stable_envs import dynamics as D

cells = json.loads(sys.argv[1])
states_dir = sys.argv[2]
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    for name in cells:
        corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)
        n = int(corpus.shape[0])
        env = snv.REGISTRY[name](num_envs=n, device=dev)
        keys = [tuple(k) for k, _, _ in env._layout()]
        env.set_state(corpus)
        st = env.state
        rec = {"device": dev, "cell": name,
               "state_keys": [".".join(k) for k in keys],
               "pack_state_equals_get_state": bool(torch.equal(env.pack(st), env.get_state())),
               "render_state_equals_render_rows": bool(
                   torch.equal(env.render(st), env.render(env.pack(st)))),
               "render_state_equals_render_held": bool(torch.equal(env.render(st), env.render())),
               "has_dynamics": bool(D.has_dynamics(env))}
        if not rec["has_dynamics"]:
            for what, args in (("step", (torch.zeros(n, 1),)),
                               ("transition", (st, torch.zeros(n, 1)))):
                try:
                    getattr(env, what)(*args)
                    rec[f"{what}_refuses"] = False
                except NotImplementedError as e:
                    rec[f"{what}_refuses"] = ("has dynamics" in str(e)
                                              and type(env).__name__ in str(e))
        else:
            g = torch.Generator().manual_seed(0)
            lo, hi = float(env.action_low), float(env.action_high)
            def acts(steps=None):
                shape = (n, env.action_dim) if steps is None else (steps, n, env.action_dim)
                return (lo + (hi - lo) * torch.rand(*shape, generator=g)).to(env.device)
            env.set_state(corpus)
            held = env.get_state().clone()
            a = acts()
            functional = env.transition(env.state, a)
            rec["transition_is_pure"] = bool(torch.equal(env.get_state(), held))
            stateful = env.step(a)["state"]
            rec["step_keys"] = list(env.step(a).keys())
            env.set_state(corpus)
            stateful = env.step(a)["state"]
            rec["transition_equals_step_bitwise"] = bool(
                torch.equal(env.pack(functional), env.pack(stateful))) and all(
                bool(torch.equal(functional[k], stateful[k])) for k in keys)
            env.set_state(corpus)
            A = acts(4)
            traj = env.unroll(env.state, A)
            cur = env.state
            ok = True
            for t in range(4):
                cur = env.transition(cur, A[t])
                ok &= all(bool(torch.equal(traj[t][k], cur[k])) for k in keys)
            rec["unroll_equals_repeated_transition"] = ok
            env.set_state(corpus)
            for t in range(4):
                env.step(A[t])
            rec["unroll_equals_four_stateful_steps"] = all(
                bool(torch.equal(traj[3][k], env.state[k])) for k in keys)
            rec["stateful_path_was_a_cuda_graph"] = bool(
                getattr(env, "_step_graph", None) is not None)
        env.close()
        rec["closed_refuses_render"] = False
        try:
            env.render()
        except RuntimeError:
            rec["closed_refuses_render"] = True
        env.close()          # idempotent
        out.append(rec)
print("@@JSON@@" + json.dumps({"checks": out}))
'''


def _run(code: str, tree: Path, python: str, *args: str) -> dict:
    env = dict(os.environ)
    extra = os.environ.get("SE9_EXTRA_PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tree}:{extra}" if extra else str(tree)
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
        if sha[:12] in block and "step9-envs-mujoco-oracle-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-envs-mujoco-oracle-baseline-{sha[:12]}"
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
    "states": "how many states were replayed",
    "roundtrip_exact": "set_state/get_state round-tripping the states",
    "rows_sha256": "the packed rows the renderer was handed",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "the official appearance",
    "registry_row": "the registry row, module included (no file in this workstream moved)",
    "sample_states_sha256": "the env's own reset distribution at a fixed seed",
    "variation_names": "the variation taxonomy, leaf for leaf",
    "parity_interior_min": "the declared interior-parity floor",
    "parity_all_exact_min": "the declared all-pixel floor",
    "fidelity_tier": "the fidelity tier",
    "render_decoder": "the decoder declaration",
}


def _functional_ok(c: dict) -> bool:
    keep = ["pack_state_equals_get_state", "render_state_equals_render_rows",
            "render_state_equals_render_held", "closed_refuses_render"]
    keep += (["step_refuses", "transition_refuses"] if not c["has_dynamics"] else
             ["transition_is_pure", "transition_equals_step_bitwise",
              "unroll_equals_repeated_transition", "unroll_equals_four_stateful_steps"])
    return all(bool(c[k]) for k in keep) and (
        c["has_dynamics"] is False or c["step_keys"] == ["state"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-states", type=int, default=N_STATES)
    #: The tree may be an rsync copy inside a clone, so ``git rev-parse HEAD`` there names the
    #: clone's HEAD and not the branch that was measured. Passed explicitly rather than guessed.
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-mujoco"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        with tempfile.TemporaryDirectory() as states_dir:
            _run(SAMPLE, baseline, a.python, json.dumps(cells), states_dir, str(a.n_states))
            before = _run(PROBE, baseline, a.python, json.dumps(cells), states_dir, a.device)
            after = _run(PROBE, ROOT, a.python, json.dumps(cells), states_dir, a.device)
            fn = _run(FUNCTIONAL, ROOT, a.python, json.dumps(cells), states_dir)

        diffs: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                if b[k] != m[k]:
                    diffs.setdefault(name, {})[k] = {"baseline": b[k], "migrated": m[k],
                                                     "means": why}

        checks = fn["checks"]
        functional_ok = all(_functional_ok(c) for c in checks)
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "compare_device": a.device,
            "workstream": "step9/envs-mujoco",
            "cells": cells,
            "states": "sampled in the BASELINE tree with sample_states(n, default_rng(0)); no cell "
                      "this workstream owns has a frozen Step-4 corpus (see the module docstring)",
            "n_states": a.n_states,
            "python": a.python,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "expected_differences": {},
            "identical": not diffs,
            "functional_checks": checks,
            "functional_ok": functional_ok,
            "baseline": before,
            "migrated": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / a.name).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

        print(json.dumps({k: record[k] for k in ("baseline_sha", "head_sha", "identical",
                                                 "unexplained_differences", "functional_ok")},
                         indent=1))
        for name in cells:
            m = record["migrated"]["cells"][name]
            print(f"  {name:30s} frames {m['frames_sha256'][:16]}  states {m['states']:3d}"
                  f"  size {m['size']}")
        ok = record["identical"] and functional_ok
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
