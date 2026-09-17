"""Oracle proof for the Workstream-B migration: the same pixels, from two different trees.

WHAT IT PROVES. The Step-8 behavioral oracle (plan §7) says the architecture may change and the
behavior may not. This is ``scratch/step9/envs-classic-nav/oracle.py``'s argument run over all
EIGHTEEN DM Control cells — the twelve fixed-camera PARITY cells and the six ``trackcom`` VISUAL
cells. Every cell's corpus is pushed through BOTH trees in two separate interpreters — a detached
``git worktree`` of the Step-8 baseline SHA and this branch, the same venv — and compared byte for
byte:

  1. the frames: ``render(states)`` at the registered default size, SHA-256 over the raw uint8 bytes,
     and again rendered from explicit packed rows;
  2. ``describe()``: the dict a consumer reads to decide whether these frames are usable;
  3. the style layout: ``style_names``, ``style_dim`` and the default (official) appearance;
  4. the registry row IN FULL, ``module`` INCLUDED — see below;
  5. the packed layout: ``state_dim``/``state_names``, and ``set_state``/``get_state`` round-tripping;
  6. ``sample_states`` at a fixed seed — the env's own reset distribution;
  7. the declarations this workstream must preserve exactly: the fidelity tier and decoder, the
     parity floors, ``visual_fidelity`` (thresholds, measurements and ``measured_at``),
     ``reference_versions``/``measured_with`` (the 3.11 provenance), the variation taxonomy leaf for
     leaf, ``RENDER_INVARIANT_DIMS``/``state_dims_note`` (Manipulator's task-specific state
     exclusions), and the CPU refusal.

THERE IS NO EXPECTED DIFFERENCE, and that is the point of a decision made in the migration rather
than discovered here. ``envs/dmcontrol.py`` became the PACKAGE ``envs/dmcontrol/`` and a module and
a package cannot share a name, so unlike Workstream A's ``swm_nav``/``navigation`` merge there is no
second import path to alias — the package IS ``stable_envs.envs.dmcontrol``. Each cell's
``__module__`` is pinned to that path (``_cells._MODULE``), so ``registry_table()[i]["module"]``,
``spec(name)["family"]`` and ``list_envs(family="dmcontrol")`` are BYTE-IDENTICAL to the baseline and
the ``module`` column is compared like every other. What IS checked separately is that every
module-level name the Step-8 module exported still resolves off the package, because that is what
the tests, the probes and ``stable_worldmodel`` reach for.

TWO CELLS HAVE NO FROZEN CORPUS (``DMCPointMassEasy``, ``DMCSwimmerSwimmer6``: neither is an SWM id
and neither was drawn into ``examples/results/step4/states/``). For those the BASELINE arm draws 64
states from its own ``sample_states(64, rng(0))``, writes them to a cache, and the migrated arm reads
THAT FILE — so the frames being compared come from one array produced by the baseline, not from two
draws that happen to agree. ``sample_states_sha256`` is compared on top, so the draw itself is pinned
too.

THE STEP-9 CLAIMS have no baseline to compare against and are checked in the migrated tree only:
every cell accepts ``num_envs=``; ``pack``/``unpack`` round-trip the packed row bit for bit;
``render(State) == render(packed rows) == render()``; and — because this family ships NO batched
simulator — ``has_dynamics`` is False and ``step``/``transition`` raise the contract's
``NotImplementedError`` naming the cell and the reason, while ``reset`` and ``render`` work.

Usage::

    MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python scratch/step9/envs-dmcontrol/oracle.py
    ... --device cuda --python $SCR/venvs/se-vis311/bin/python   # the CUDA arm
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

#: The twelve PARITY cells and the six VISUAL ones, in registry order.
PARITY_CELLS = ("DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
                "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard",
                "DMCManipulatorBringBall", "DMCPendulumSwingup", "DMCPointMassEasy",
                "DMCReacherEasy", "DMCReacherHard")
TRACKCOM_CELLS = ("DMCCheetahRun", "DMCHopperHop", "DMCHumanoidStand", "DMCQuadrupedWalk",
                  "DMCSwimmerSwimmer6", "DMCWalkerWalk")
CELLS = PARITY_CELLS + TRACKCOM_CELLS

#: Where each cell's frozen corpus lives, relative to the tree root. The two Step-7 cells were
#: certified after Step 4 froze its corpus directory and carry theirs beside their findings; the two
#: absent ones are drawn once by the baseline arm and cached (see the module docstring).
CORPUS = {
    "DMCManipulatorBringBall": "examples/results/step7/dmc-manipulator/corpus/states/"
                               "DMCManipulatorBringBall.npz",
    "DMCQuadrupedWalk": "examples/results/step7/dmc-quadruped/phase1/cell/states.npz",
}
DRAWN = ("DMCPointMassEasy", "DMCSwimmerSwimmer6")

#: The one Step-8 module-level name the package deliberately does NOT carry, with the reason. It was
#: the private factory that built all eighteen classes from two tables in one ``type()`` loop; the
#: cells are now written where their domain is and there is no loop to name. Nothing outside the
#: module ever referenced it (checked: ``grep -rn _make_classes`` finds only its definition and its
#: one call), and it is recorded here rather than silently tolerated so that a name that vanished
#: for a worse reason would still fail.
EXPECTED_ABSENT = {
    "_make_classes": "the private factory that built the eighteen classes from two tables; the "
                     "cells are now declared in their own domain modules, so there is no loop to "
                     "name. Never referenced outside the module.",
}

#: Run inside ONE tree, printing a JSON record on stdout. Written as a string and executed with
#: ``PYTHONPATH`` pointed at the tree under test, because the two trees define the same module names
#: and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, os, sys
import numpy as np, torch

import stable_envs as se

cells = json.loads(sys.argv[1])
corpus_rel = json.loads(sys.argv[2])
cache = sys.argv[3]
dev = sys.argv[4]
tree = sys.argv[5]

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

try:
    from stable_envs.env import reference_versions_of as _refver
except Exception:
    from stable_envs.core.env import reference_versions_of as _refver

def jsonable(x):
    return json.loads(json.dumps(x, default=str))

table = {r["name"]: r for r in se.registry_table()}
out = {"cells": {}, "import_errors": (se.import_errors() if hasattr(se, "import_errors")
                                      else se.envs.import_errors())}
for name in cells:
    rec = {"device": dev}
    cls = se.REGISTRY.get(name)
    if cls is None:
        out["cells"][name] = {"unregistered": True}
        continue
    rel = corpus_rel.get(name)
    cached = os.path.join(cache, name + ".npz")
    try:
        try:
            env = cls(num_envs=64, device=dev)
        except TypeError as exc:                  # Step 8 spelled it ``batch=``
            if "num_envs" not in str(exc):
                raise
            env = cls(batch=64, device=dev)
    except Exception as exc:
        # A cell that REFUSES this device is a result, not a failure: the refusal and its message
        # are compared between the two trees like every other declaration.
        out["cells"][name] = {"device": dev,
                              "construct_refused": f"{type(exc).__name__}: {exc}"}
        continue
    try:
        if rel is not None:
            corpus = np.load(os.path.join(tree, rel))["states"].astype(np.float32)
        elif os.path.exists(cached):
            corpus = np.load(cached)["states"].astype(np.float32)
        else:                                  # the baseline arm draws it once and caches it
            corpus = np.asarray(env.sample_states(64, np.random.default_rng(0)), np.float32)
            os.makedirs(cache, exist_ok=True)
            np.savez(cached, states=corpus)
        n = int(corpus.shape[0])
        if n != 64:
            env.close()
            env = (cls(num_envs=n, device=dev))
        env.set_state(corpus)
        rows = env.get_state().detach().cpu().numpy()
        rec.update({
            "state_dim": int(env.state_dim),
            "state_names": list(env.state_names),
            "size": int(env.size),
            "default_supersample": int(getattr(cls, "default_supersample", 1)),
            "corpus_states": n,
            "corpus_sha256": h(corpus),
            "roundtrip_exact": bool(np.array_equal(rows, corpus)),
            "rows_sha256": h(rows),
            "describe": jsonable(env.describe()),
            "style_names": list(env.style_names),
            "style_dim": int(env.style_dim),
            "default_style_sha256": h(np.asarray(env.default_style(), np.float32)),
            "registry_row": table[name],
            "sample_states_sha256": h(np.asarray(env.sample_states(16, np.random.default_rng(0)),
                                                 np.float32)),
            "variation_names": list(env.variation_space.names()),
            "parity_interior_min": getattr(env, "parity_interior_min", None),
            "parity_all_exact_min": getattr(env, "parity_all_exact_min", None),
            "parity_note": str(getattr(cls, "parity_note", "") or ""),
            "fidelity_tier": env.fidelity_tier,
            "render_decoder": env.render_decoder,
            "visual_fidelity": jsonable(getattr(cls, "visual_fidelity", {}) or {}),
            "visual_note": str(getattr(cls, "visual_note", "") or ""),
            "reference_versions": jsonable(_refver(cls)),
            "reference_id": getattr(cls, "reference_id", None),
            "render_invariant_dims": list(getattr(cls, "RENDER_INVARIANT_DIMS", ()) or ()),
            "state_dims_note": str(getattr(cls, "state_dims_note", "") or ""),
            "shape_knobs": jsonable(getattr(cls, "SHAPE_KNOBS", {}) or {}),
            "domain_task_camera": [cls.DOMAIN, cls.TASK, int(cls.CAMERA)],
            "docstring": (cls.__doc__ or ""),
        })
        try:
            frames = env.render().detach().cpu().numpy()
            rec["frames_shape"] = list(frames.shape)
            rec["frames_sha256"] = h(frames)
            rec["frames_mean"] = float(frames.mean())
            fr = env.render(torch.as_tensor(corpus)).detach().cpu().numpy()
            rec["frames_from_rows_sha256"] = h(fr)
        except Exception as exc:
            rec["render_refused"] = f"{type(exc).__name__}: {exc}"
    finally:
        if hasattr(env, "close"):            # Step 8 had no close(); the baseline arm runs there too
            env.close()
    out["cells"][name] = rec

# Every module-level name the Step-8 module exported must still resolve off the package.
from stable_envs.envs import dmcontrol as D
wanted = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
out["module_names_resolve"] = {n: hasattr(D, n) for n in wanted}
out["module_is_package"] = hasattr(D, "__path__")
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the migrated tree only (there is nothing to compare them against).
FUNCTIONAL = r'''
import json, os, sys
import numpy as np, torch
import stable_envs as snv

cells = json.loads(sys.argv[1])
corpus_rel = json.loads(sys.argv[2])
cache = sys.argv[3]
tree = sys.argv[4]
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    for name in cells:
        cls = getattr(snv, name)
        rec = {"device": dev, "cell": name}
        try:
            env = cls(num_envs=16, device=dev)          # contract §2: num_envs= on every cell
        except ValueError as exc:                       # a declared device refusal, e.g. swimmer
            rec["construct_refused"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        except Exception as exc:
            rec["unconstructible"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        try:
            rel = corpus_rel.get(name)
            cached = os.path.join(cache, name + ".npz")
            if rel is not None:
                corpus = np.load(os.path.join(tree, rel))["states"].astype(np.float32)[:16]
            elif os.path.exists(cached):
                corpus = np.load(cached)["states"].astype(np.float32)[:16]
            else:
                corpus = np.asarray(env.sample_states(16, np.random.default_rng(0)), np.float32)
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
                rec["render_state_equals_render_rows"] = bool(torch.equal(a, b)
                                                              and torch.equal(a, c))
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
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-6000:]}\n{r.stderr[-6000:]}")
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
    "default_supersample": "the shipped supersample every parity number was measured at",
    "corpus_states": "how many frozen states were replayed",
    "corpus_sha256": "the states themselves — both arms must replay ONE array",
    "roundtrip_exact": "set_state/get_state round-tripping the corpus",
    "rows_sha256": "the packed rows the renderer was handed",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "render_refused": "whether this cell renders on this device, and the refusal if not",
    "construct_refused": "the CPU refusal this cell declares, message included",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "the official default appearance",
    "registry_row": "the registry row IN FULL, module included",
    "sample_states_sha256": "the env's own reset distribution at a fixed seed",
    "variation_names": "the variation taxonomy, leaf for leaf",
    "parity_interior_min": "the declared interior-parity floor",
    "parity_all_exact_min": "the declared all-pixel floor",
    "parity_note": "the prose the declared floor rests on",
    "fidelity_tier": "the fidelity tier",
    "render_decoder": "the decoder declaration",
    "visual_fidelity": "the measured numbers AND the thresholds of a visual cell",
    "visual_note": "the named residual mechanisms a visual cell declares",
    "reference_versions": "the mujoco 3.11 certification provenance",
    "reference_id": "what this cell is certified against",
    "render_invariant_dims": "Manipulator's task-specific state exclusions",
    "state_dims_note": "the contract those exclusions are granted under",
    "shape_knobs": "the discrete shape alternatives a cell offers",
    "domain_task_camera": "which dm_control scene this cell IS",
    "docstring": "the class docstring, which carries the tier claim a user reads",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--cache", default=None,
                    help="where the two drawn corpora are cached; the baseline arm writes, the "
                         "migrated arm reads")
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-dmcontrol"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]
    cache = a.cache or str(Path(tempfile.gettempdir()) / "step9-dmc-oracle-corpus")

    # Every module-level name the Step-8 module exported, read off the baseline SOURCE so the list
    # cannot drift from what that file actually had.
    import ast
    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        src = (baseline / "stable_envs/envs/dmcontrol.py").read_text()
        wanted = []
        for node in ast.parse(src).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                wanted.append(node.name)
            elif isinstance(node, ast.Assign):
                wanted += [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                wanted.append(node.target.id)
        wanted = sorted(set(wanted) - {"annotations"})

        aj = json.dumps
        before = _run(PROBE, baseline, a.python, aj(cells), aj(CORPUS), cache, a.device,
                      str(baseline), aj(wanted))
        after = _run(PROBE, ROOT, a.python, aj(cells), aj(CORPUS), cache, a.device,
                     str(ROOT), aj(wanted))
        fn = _run(FUNCTIONAL, ROOT, a.python, aj(cells), aj(CORPUS), cache, str(ROOT))

        diffs: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                bv, mv = b.get(k, "<absent>"), m.get(k, "<absent>")
                if bv != mv:
                    diffs.setdefault(name, {})[k] = {"baseline": bv, "migrated": mv, "means": why}

        missing = sorted(n for n, ok in after["module_names_resolve"].items()
                         if not ok and n not in EXPECTED_ABSENT)
        absent = sorted(n for n, ok in after["module_names_resolve"].items()
                        if not ok and n in EXPECTED_ABSENT)
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
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "compare_device": a.device,
            "workstream": "step9/envs-dmcontrol",
            "cells": cells,
            "corpus": {n: CORPUS.get(n, "drawn once by the baseline arm and cached") for n in cells},
            "python": a.python,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "expected_differences": {n: EXPECTED_ABSENT[n] for n in absent},
            "identical": not diffs,
            "step8_module_names_checked": len(wanted),
            "step8_module_names_missing": missing,
            "step8_module_names_expected_absent": absent,
            "module_is_package": after["module_is_package"],
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
            "step8_module_names_missing", "expected_differences",
            "functional_ok")}, indent=1)[:4000])
        for name in cells:
            m = record["migrated"]["cells"][name]
            print(f"  {name:26s} frames {str(m.get('frames_sha256'))[:16]}"
                  f"  states {m.get('corpus_states')}"
                  + (f"  RENDER REFUSED: {m['render_refused'][:60]}" if "render_refused" in m
                     else ""))
        ok = record["identical"] and functional_ok and not missing
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
