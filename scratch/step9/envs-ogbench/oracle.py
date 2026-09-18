"""Oracle proof for the Workstream-D migration: the same pixels and the same declarations, from two trees.

WHAT IT PROVES. The Step-8 behavioral oracle (plan §7) says the architecture may change and the
behavior may not. This is ``scratch/step9/envs-dmcontrol/oracle.py``'s argument run over the OGBench
family — the 25 maze cells (11 distinct scenes, ``parity`` tier, CPU-capable) and the two
manipulation cells ``OGBCube``/``OGBScene`` (``visual`` tier, CUDA-only by declaration). Every cell
is pushed through BOTH trees in two separate interpreters — a detached ``git worktree`` of the
Step-8 baseline SHA and this branch, the same venv — and compared:

  1. THE FRAMES for the mazes: ``render(states)`` at the registered default size on ``cpu`` (the
     raycast3d CPU path), SHA-256 over the raw uint8 bytes, and again from explicit packed rows. The
     frozen corpora are ``examples/results/step4/states/*maze*.npz``: eleven files, one per DISTINCT
     SCENE, and the fourteen cells that have no file of their own are replayed on their scene
     sibling's corpus, because ``navigate``/``stitch``/``explore`` select a dataset and never reach
     the renderer — so a sibling's states are exactly the states of this cell's scene;
  2. ``describe()``, the style layout (``style_names``, ``style_dim``, the official default
     appearance) and the variation taxonomy leaf for leaf — including OGBMaze's
     ``rendering.decolor_floor``, whose SWM counterpart ``floor.pixel_encoding`` is
     UNSUPPORTED-PENDING in the variation map;
  3. the registry row IN FULL, ``module`` INCLUDED, plus the SWM id aliases the registry resolves for
     these cells (``swm/OGBMaze-v0``, ``swm/OGBCube-v0``, ``swm/OGBScene-v0``);
  4. ``state_dim``/``state_names`` and the ``set_state``/``get_state`` round trip on the corpus;
  5. ``sample_states`` at a fixed seed — the env's own reset distribution;
  6. the declarations this workstream must preserve exactly: the fidelity tier and decoder, the
     parity floor and ``parity_note``, ``visual_fidelity`` in full (measurements AND thresholds),
     ``visual_note`` (which carries the button-texture bake note), ``reference_versions`` /
     ``measured_with``, ``default_supersample`` and the class docstring;
  7. THE CPU REFUSAL for the two manipulation cells, message included: they cannot be constructed on
     ``cpu`` at all, and the ``ValueError`` naming ``CPU3D-OGB-1`` is compared like any other string.
     Their class-level declarations and their variation space are compared WITHOUT constructing
     (``make_variation_space`` needs only the cached bundle, which is mujoco + ogbench + swm on the
     CPU), which is how a CUDA-only cell is audited on a Mac.

OWED TO THE GPU ORACLE, and recorded rather than claimed: the FRAMES of ``OGBCube`` and
``OGBScene``. Those cells have no CPU backend by declaration, so there is no honest way to render
them here; ``examples/results/step9/envs-ogbench/findings.md`` lists them under "owed", and running
this same script with ``--device cuda --cells OGBCube,OGBScene`` on a CUDA host is the whole of what
is missing.

THE EXPECTED DIFFERENCE IS EXACTLY ONE and it is declared here rather than discovered in the diff:
``stable_envs/envs/ogbench.py`` became the PACKAGE ``stable_envs/envs/ogbench/``, and a module and a
package cannot share a name. So this script also reads every module-level name off the BASELINE
SOURCE of both Step-8 modules and checks each one still resolves — off the package for the maze
half, off the ``sys.modules`` alias ``stable_envs.envs.ogb_manip`` for the manipulation half.

THE STEP-9 CLAIMS have no baseline to compare against and are checked in the migrated tree only:
every cell accepts ``num_envs=``; ``pack``/``unpack`` round-trip the packed row bit for bit;
``pack(env.state) == get_state()``; ``render(State) == render(rows) == render()``; and — because
this family ships NO batched simulator — ``has_dynamics`` is False and ``step``/``transition`` raise
the contract's ``NotImplementedError``, while ``reset`` and ``render`` work and a closed env refuses.

Usage::

    MUJOCO_GL=glfw /Users/samib/se7-ogb-venv/bin/python scratch/step9/envs-ogbench/oracle.py
    ... --device cuda --python $SCR/venvs/se-step5/bin/python    # the owed GPU arm
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: The 25 maze cells, as ``(loco, maze, dataset)`` is spelled in ``maze.CELLS``.
_MAZE = (
    [("point", m, d) for m in ("medium", "large", "giant", "teleport") for d in ("navigate", "stitch")]
    + [("ant", m, d) for m in ("medium", "large") for d in ("navigate", "stitch", "explore")]
    + [("ant", "giant", d) for d in ("navigate", "stitch")]
    + [("ant", "teleport", d) for d in ("navigate", "stitch", "explore")]
    + [("humanoid", m, d) for m in ("medium", "large", "giant") for d in ("navigate", "stitch")]
)
MAZE_CELLS = tuple(f"{lo.capitalize()}maze{mz.capitalize()}{ds.capitalize()}" for lo, mz, ds in _MAZE)
MANIP_CELLS = ("OGBCube", "OGBScene")
CELLS = MAZE_CELLS + MANIP_CELLS

#: The frozen Step-4 corpus each cell is replayed on. There are ELEVEN files for 25 cells, one per
#: distinct SCENE; the other fourteen share a scene with one of them and are replayed on its states,
#: because the dataset suffix selects a task and the task never reaches the renderer. That is the
#: same fact the plate cache is keyed on, so replaying a sibling's corpus is not a shortcut — it is
#: the only thing there is to replay.
_FROZEN = {  # (loco, maze) -> the cell whose corpus is committed
    ("point", "medium"): "PointmazeMediumNavigate", ("point", "large"): "PointmazeLargeNavigate",
    ("point", "giant"): "PointmazeGiantNavigate", ("point", "teleport"): "PointmazeTeleportNavigate",
    ("ant", "medium"): "AntmazeMediumExplore", ("ant", "large"): "AntmazeLargeExplore",
    ("ant", "giant"): "AntmazeGiantNavigate", ("ant", "teleport"): "AntmazeTeleportExplore",
    ("humanoid", "medium"): "HumanoidmazeMediumNavigate",
    ("humanoid", "large"): "HumanoidmazeLargeNavigate",
    ("humanoid", "giant"): "HumanoidmazeGiantNavigate",
}
CORPUS = {f"{lo.capitalize()}maze{mz.capitalize()}{ds.capitalize()}":
          f"examples/results/step4/states/{_FROZEN[(lo, mz)]}.npz" for lo, mz, ds in _MAZE}

#: Cells whose FRAMES this script cannot compare on a CPU host, with the reason. They are not
#: silently skipped: the record carries them under ``owed_to_the_gpu_oracle`` and the run is not
#: called a pass for them, only for what it did compare.
OWED_ON_CPU = {
    "OGBCube": "no CPU backend by declaration (CPU3D-OGB-1); frames are owed to the GPU oracle",
    "OGBScene": "no CPU backend by declaration (CPU3D-OGB-1); frames are owed to the GPU oracle",
}

#: Step-8 module-level names that the migrated tree deliberately does not carry, with the reason.
#: Empty: every name of BOTH Step-8 modules still resolves. Kept as a declared-exceptions slot so a
#: name that vanished for a worse reason would still fail the run.
EXPECTED_ABSENT: dict[str, str] = {}

#: Run inside ONE tree, printing a JSON record on stdout. Written as a string and executed with
#: ``PYTHONPATH`` pointed at the tree under test, because the two trees define the same module names
#: and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, os, sys, types
import numpy as np, torch

import stable_envs as se

cells = json.loads(sys.argv[1])
corpus_rel = json.loads(sys.argv[2])
dev = sys.argv[3]
tree = sys.argv[4]
maze_names = json.loads(sys.argv[5])
manip_names = json.loads(sys.argv[6])

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

try:
    from stable_envs.env import reference_versions_of as _refver
except Exception:
    from stable_envs.core.env import reference_versions_of as _refver

def jsonable(x):
    return json.loads(json.dumps(x, default=str))

def space_record(space):
    """A variation space as data: the taxonomy, the flat layout and every leaf's init value."""
    lay = [[p, int(o), int(w)] for p, o, w in space.flat_layout()]
    inits = {}
    for p, _, _ in space.flat_layout():
        node = space
        for part in p.split("."):
            node = node[part]
        inits[p] = jsonable(np.asarray(node.value).reshape(-1).tolist())
    return {"names": list(space.names()), "flat_layout": lay, "flat_dim": int(space.flat_dim),
            "init_values": inits}

table = {r["name"]: r for r in se.registry_table()}
out = {"cells": {}}
for name in cells:
    rec = {"device": dev}
    cls = se.REGISTRY.get(name)
    if cls is None:
        out["cells"][name] = {"unregistered": True}
        continue
    # ---- everything that is a CLASS declaration, readable without constructing anything -------
    rec.update({
        "registry_row": table.get(name),
        "reference_id": getattr(cls, "reference_id", None),
        "default_size": int(getattr(cls, "default_size", 0) or 0),
        "default_supersample": int(getattr(cls, "default_supersample", 1)),
        "fidelity_tier": getattr(cls, "fidelity_tier", None),
        "render_decoder": getattr(cls, "render_decoder", None),
        "parity_interior_min": getattr(cls, "parity_interior_min", None),
        "parity_all_exact_min": getattr(cls, "parity_all_exact_min", None),
        "parity_note": str(getattr(cls, "parity_note", "") or ""),
        "visual_fidelity": jsonable(getattr(cls, "visual_fidelity", {}) or {}),
        "visual_note": str(getattr(cls, "visual_note", "") or ""),
        "reference_versions": jsonable(_refver(cls)),
        "docstring": (cls.__doc__ or ""),
        "class_module": cls.__module__,
        "loco_maze": [getattr(cls, "LOCO", None), getattr(cls, "MAZE", None)],
        "cell_tag": getattr(cls, "CELL", None),
        "tile": getattr(cls, "TILE", None),
        "plate_paths": list(getattr(cls, "PLATE_PATHS", ()) or ()),
    })
    # ---- the variation space, which a CUDA-only cell can still answer for ---------------------
    try:
        if getattr(cls, "CELL", None):     # the manipulation cells: unbound, on a CELL-only stub
            rec["variation"] = space_record(
                cls.make_variation_space(types.SimpleNamespace(CELL=cls.CELL)))
        else:
            rec["variation"] = space_record(
                cls.make_variation_space(types.SimpleNamespace(LOCO=cls.LOCO, MAZE=cls.MAZE)))
    except Exception as exc:
        rec["variation_failed"] = f"{type(exc).__name__}: {exc}"
    # ---- construct, if this device serves this cell -------------------------------------------
    try:
        try:
            env = cls(num_envs=64, device=dev)
        except TypeError as exc:                  # Step 8 spelled it ``batch=``
            if "num_envs" not in str(exc):
                raise
            env = cls(batch=64, device=dev)
    except Exception as exc:
        # A cell that REFUSES this device is a result, not a failure: the refusal and its message
        # are compared between the two trees like every other declaration. This is the CPU refusal.
        rec["construct_refused"] = f"{type(exc).__name__}: {exc}"
        out["cells"][name] = rec
        continue
    try:
        corpus = np.load(os.path.join(tree, corpus_rel[name]))["states"].astype(np.float32)
        n = int(corpus.shape[0])
        if n != 64:
            env.close()
            env = cls(num_envs=n, device=dev)
        env.set_state(corpus)
        rows = env.get_state().detach().cpu().numpy()
        rec.update({
            "state_dim": int(env.state_dim),
            "state_names": list(env.state_names),
            "size": int(env.size),
            "supersample": int(env.ss),
            "corpus": corpus_rel[name],
            "corpus_states": n,
            "corpus_sha256": h(corpus),
            "roundtrip_exact": bool(np.array_equal(rows, corpus)),
            "rows_sha256": h(rows),
            "describe": jsonable(env.describe()),
            "style_names": list(env.style_names),
            "style_dim": int(env.style_dim),
            "default_style_sha256": h(np.asarray(env.default_style(), np.float32)),
            "variation_names": list(env.variation_space.names()),
            "sample_states_sha256": h(np.asarray(env.sample_states(16, np.random.default_rng(0)),
                                                 np.float32)),
        })
        try:
            frames = env.render().detach().cpu().numpy()
            rec["frames_shape"] = list(frames.shape)
            rec["frames_sha256"] = h(frames)
            rec["frames_mean"] = round(float(frames.mean()), 6)
            fr = env.render(torch.as_tensor(corpus)).detach().cpu().numpy()
            rec["frames_from_rows_sha256"] = h(fr)
        except Exception as exc:
            rec["render_refused"] = f"{type(exc).__name__}: {exc}"
    finally:
        if hasattr(env, "close"):            # Step 8 had no close(); the baseline arm runs there too
            env.close()
    out["cells"][name] = rec

# THE SWM ID ALIASES, read from the source BOTH trees have: the VariationMap tables, where every
# cell's `swm_id` is declared. The registry's own `SWM_IDS` table is a Step-9 addition (the Wave-0
# alias table), so asking for it in the baseline arm would compare a thing against its own absence;
# it is recorded separately below, from the migrated tree only, and checked against this.
out["swm_ids"] = {}
try:
    from stable_envs.compat.variation_maps.newcells import NEWCELL_VARIATION_MAPS as _VM
    for name in cells:
        vm = _VM.get(name)
        out["swm_ids"][name] = None if vm is None else vm.swm_id
except Exception as exc:
    out["swm_ids"] = f"{type(exc).__name__}: {exc}"
# Migrated-tree only, recorded and not compared: what the Step-9 registry resolves each id to.
try:
    from stable_envs.registry import SWM_IDS as _SID
    out["registry_swm_ids"] = {k: (sorted(v) if isinstance(v, (list, tuple, set)) else v)
                               for k, v in _SID.items()
                               if k in ("swm/OGBMaze-v0", "swm/OGBCube-v0", "swm/OGBScene-v0")}
except Exception as exc:
    out["registry_swm_ids"] = f"{type(exc).__name__}: {exc}"

# Every module-level name the two Step-8 modules exported must still resolve.
import importlib
mods = {}
try:
    mods["maze"] = importlib.import_module("stable_envs.envs.ogbench")
except Exception as exc:
    mods["maze"] = None
    out["maze_import_error"] = f"{type(exc).__name__}: {exc}"
try:
    mods["manip"] = importlib.import_module("stable_envs.envs.ogb_manip")
except Exception as exc:
    mods["manip"] = None
    out["manip_import_error"] = f"{type(exc).__name__}: {exc}"
out["module_names_resolve"] = {
    "maze": {n: (mods["maze"] is not None and hasattr(mods["maze"], n)) for n in maze_names},
    "manip": {n: (mods["manip"] is not None and hasattr(mods["manip"], n)) for n in manip_names},
}
out["maze_is_package"] = bool(mods["maze"] is not None and hasattr(mods["maze"], "__path__"))
out["cache_objects_are_shared"] = bool(
    mods["maze"] is not None
    and mods["maze"]._SCENE_CACHE is importlib.import_module(
        mods["maze"].OGBenchMaze.__module__ if mods["maze"].OGBenchMaze.__module__
        != "stable_envs.envs.ogbench" else "stable_envs.envs.ogbench")._SCENE_CACHE)
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the migrated tree only (there is nothing to compare them against).
FUNCTIONAL = r'''
import json, os, sys
import numpy as np, torch
import stable_envs as snv

cells = json.loads(sys.argv[1])
corpus_rel = json.loads(sys.argv[2])
tree = sys.argv[3]
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    for name in cells:
        cls = getattr(snv, name)
        rec = {"device": dev, "cell": name}
        try:
            env = cls(num_envs=16, device=dev)          # contract §2: num_envs= on every cell
        except ValueError as exc:                       # a declared device refusal (CPU3D-OGB-1)
            rec["construct_refused"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        except Exception as exc:
            rec["unconstructible"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        try:
            corpus = np.load(os.path.join(tree, corpus_rel[name]))["states"].astype(np.float32)[:16]
            env.set_state(corpus)
            rows = env.get_state()
            st = env.state
            rec["num_envs_accepted"] = bool(len(env) == 16 and env.num_envs == 16)
            rec["state_keys"] = [".".join(k) for k, _, _ in env._layout()]
            rec["pack_unpack_roundtrip"] = bool(torch.equal(env.pack(env.unpack(rows)), rows))
            rec["pack_state_is_get_state"] = bool(torch.equal(env.pack(st), rows))
            rec["no_hidden_fields"] = all(o is not None for _, _, o in env._layout())
            rec["has_dynamics"] = bool(env.has_dynamics)
            for call in ("step", "transition"):
                try:
                    if call == "step":
                        env.step(torch.zeros(16, 1, device=env.device))
                    else:
                        env.transition(st, torch.zeros(16, 1, device=env.device))
                    rec[call + "_refused"] = False
                except NotImplementedError as exc:
                    rec[call + "_refused"] = True
                    rec[call + "_message"] = str(exc)
                except Exception as exc:
                    rec[call + "_refused"] = False
                    rec[call + "_wrong_error"] = f"{type(exc).__name__}: {exc}"
            try:
                a = env.render(st)
                b = env.render(env.pack(st))
                c = env.render()
                rec["render_state_equals_render_rows"] = bool(torch.equal(a, b) and torch.equal(a, c))
            except Exception as exc:
                rec["render_refused"] = f"{type(exc).__name__}: {exc}"
            rec["reset_works"] = bool(env.reset(seed=0).batch_size[0] == 16)
        finally:
            env.close()
        rec["closed_refuses_render"] = False
        try:
            env.render()
        except RuntimeError:
            rec["closed_refuses_render"] = True
        out.append(rec)
print("@@JSON@@" + json.dumps({"checks": out}))
'''


def _run(code: str, tree: Path, python: str, *args: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", code, *args], capture_output=True, text=True,
                       cwd=str(tree), env=env)
    if r.returncode != 0:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-8000:]}\n{r.stderr[-8000:]}")
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


def _module_level_names(src: str) -> list[str]:
    """Every module-level name a Step-8 source file bound, read off its AST rather than imported."""
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.Assign):
            out += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
    return sorted(set(out) - {"annotations"})


#: Every key the two trees must agree on, and the sentence that says what a difference would mean.
COMPARED = {
    "device": "the device the comparison arm ran on",
    "registry_row": "the registry row IN FULL, module included",
    "reference_id": "what this cell is certified against",
    "default_size": "the registered default frame size",
    "default_supersample": "the shipped supersample every parity number was measured at",
    "fidelity_tier": "the fidelity tier",
    "render_decoder": "the decoder declaration",
    "parity_interior_min": "the declared interior-parity floor (NOT relaxed: 1.0 on every maze)",
    "parity_all_exact_min": "the declared all-pixel floor",
    "parity_note": "the agent-pixel residual statement the floor rests on",
    "visual_fidelity": "the measured numbers AND the thresholds of a visual cell",
    "visual_note": "the named residual mechanisms, including the Scene's button-texture bake",
    "reference_versions": "the certification provenance",
    "docstring": "the class docstring, which carries the tier claim a user reads",
    "class_module": "__module__, which is where registry.spec reads the family from",
    "loco_maze": "which OGBench scene a maze cell IS",
    "cell_tag": "which swm scene a manipulation cell IS",
    "tile": "the screen-space window the movers are cast in",
    "plate_paths": "which style leaves re-bake the plate",
    "variation": "the variation taxonomy, the flat layout and every leaf's init value",
    "variation_failed": "whether the variation space could be built at all",
    "construct_refused": "the CPU refusal this cell declares, message included",
    "state_dim": "the packed row's width",
    "state_names": "the packed row's column names",
    "size": "the frame size the constructed env uses",
    "supersample": "the supersample the constructed env uses",
    "corpus": "which frozen file was replayed",
    "corpus_states": "how many frozen states were replayed",
    "corpus_sha256": "the states themselves — both arms must replay ONE array",
    "roundtrip_exact": "set_state/get_state round-tripping the corpus",
    "rows_sha256": "the packed rows the renderer was handed",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_mean": "the frames' mean, so a digest miss says how far apart they are",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "render_refused": "whether this cell renders on this device, and the refusal if not",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "the official default appearance",
    "variation_names": "the variation taxonomy of the CONSTRUCTED env, leaf for leaf",
    "sample_states_sha256": "the env's own reset distribution at a fixed seed",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-ogbench"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        maze_names = _module_level_names((baseline / "stable_envs/envs/ogbench.py").read_text())
        manip_names = _module_level_names((baseline / "stable_envs/envs/ogb_manip.py").read_text())

        aj = json.dumps
        args = (aj(cells), aj(CORPUS), a.device)
        before = _run(PROBE, baseline, a.python, *args, str(baseline), aj(maze_names),
                      aj(manip_names))
        after = _run(PROBE, ROOT, a.python, *args, str(ROOT), aj(maze_names), aj(manip_names))
        fn = _run(FUNCTIONAL, ROOT, a.python, aj(cells), aj(CORPUS), str(ROOT))

        diffs: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                bv, mv = b.get(k, "<absent>"), m.get(k, "<absent>")
                if bv != mv:
                    diffs.setdefault(name, {})[k] = {"baseline": bv, "migrated": mv, "means": why}
        if before["swm_ids"] != after["swm_ids"]:
            diffs.setdefault("<registry>", {})["swm_ids"] = {
                "baseline": before["swm_ids"], "migrated": after["swm_ids"],
                "means": "the SWM id alias each cell serves, per the VariationMap tables"}
        # The Step-9 registry's own alias table must agree with those declarations, cell for cell.
        reg = after.get("registry_swm_ids") or {}
        if isinstance(reg, dict):
            for sid, names in reg.items():
                want = sorted(n for n, s in after["swm_ids"].items() if s == sid)
                if want and sorted(names) != want:
                    diffs.setdefault("<registry>", {})[f"registry {sid}"] = {
                        "baseline": want, "migrated": sorted(names),
                        "means": "the registry's SWM alias table against the cells' own swm_id"}

        missing = {half: sorted(n for n, ok in after["module_names_resolve"][half].items()
                                if not ok and n not in EXPECTED_ABSENT)
                   for half in ("maze", "manip")}
        checks = fn["checks"]
        functional_ok = all(
            c.get("unconstructible") is None
            and (c.get("construct_refused") is not None or (
                c.get("num_envs_accepted") and c.get("pack_unpack_roundtrip")
                and c.get("pack_state_is_get_state") and c.get("no_hidden_fields")
                and c.get("has_dynamics") is False
                and c.get("step_refused") and c.get("transition_refused")
                and c.get("reset_works") and c.get("closed_refuses_render")
                and (c.get("render_state_equals_render_rows") or c.get("render_refused"))))
            for c in checks)
        framed = sorted(n for n in cells if "frames_sha256" in after["cells"][n])
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "compare_device": a.device,
            "workstream": "step9/envs-ogbench",
            "cells": cells,
            "python": a.python,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "expected_differences": dict(EXPECTED_ABSENT),
            "identical": not diffs,
            "cells_with_frames_compared": framed,
            "owed_to_the_gpu_oracle": {n: OWED_ON_CPU[n] for n in cells
                                       if n in OWED_ON_CPU and n not in framed},
            "step8_module_names_checked": {"maze": len(maze_names), "manip": len(manip_names)},
            "step8_module_names_missing": missing,
            "maze_is_package": after["maze_is_package"],
            "cache_objects_are_shared": after["cache_objects_are_shared"],
            "functional_checks": checks,
            "functional_ok": functional_ok,
            "baseline": before,
            "migrated": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / a.name).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

        print(json.dumps({k: record[k] for k in (
            "baseline_sha", "head_sha", "compare_device", "identical",
            "unexplained_differences", "step8_module_names_checked",
            "step8_module_names_missing", "owed_to_the_gpu_oracle", "maze_is_package",
            "cache_objects_are_shared", "functional_ok")}, indent=1)[:6000])
        for name in cells:
            m = record["migrated"]["cells"][name]
            print(f"  {name:28s} frames {str(m.get('frames_sha256'))[:16]}"
                  f"  states {m.get('corpus_states')}"
                  + (f"  REFUSED: {m['construct_refused'][:52]}" if "construct_refused" in m
                     else ""))
        ok = (record["identical"] and functional_ok
              and not missing["maze"] and not missing["manip"])
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
