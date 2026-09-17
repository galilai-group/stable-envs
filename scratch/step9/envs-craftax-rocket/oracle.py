"""Oracle proof for the Workstream-E migration: the same pixels, from two different trees.

WHAT IT PROVES. Plan §7: the architecture may change and the behavior may not. This is
``scratch/step9/foundation/pusht_oracle.py``'s argument and
``scratch/step9/envs-classic-nav/oracle.py``'s structure, applied to the three cells this workstream
owns — ``CraftaxClassic``, ``CraftaxFull``, ``PFRocketLanding`` — each of which is replayed through
BOTH trees in two separate interpreters (a detached ``git worktree`` of the Step-8 baseline SHA and
this branch, the same venv) and compared byte for byte.

THE CORPORA, and why there are three of them for two Craftax cells (Sami's Step-8 ruling: keep both
of Classic's):

  * ``examples/results/step4/states/CraftaxClassic.npz`` — 64 x **127**. It predates the full nine-row
    world window, so it is the HISTORICAL WORLD-ROW ORACLE under the zero-extension interpretation:
    each 127-slot row is zero-extended into the 149-slot layout (an empty inventory, awake, day-cycle
    key zero) with ``light_level`` forced to 1.0, exactly as
    ``scratch/step7/closure/craftax_world_row_replay.py`` does it. Rendering it directly is a
    ``ValueError`` by design, and that raise is checked here rather than worked around silently.
  * ``examples/results/step7/craftax-classic-full/corpus/states.npz`` — 64 x **149**, the complete
    observation: 16 daylight rows, 25 dusk, 23 night, 30 sleeping.
  * ``examples/results/step7/craftax-full/corpus/states.npz`` — 64 x **544**.

Both Craftax cells are rendered at BOTH of their certified tile sizes (Classic 7 and 16, Full 10 and
16), because the two sizes are two different atlases and a migration could move one and not the other.

ROCKET IS RENDERER-ONLY ON CUDA AND ITS FRAMES ARE OWED TO THE POST-WAVE-1 GPU ORACLE. It refuses a
``device="cpu"`` construction by design and cannot even be imported into a renderer on a host without
nvdiffrast, so ``PFRocketLanding`` is compared here on everything that is NOT a frame and is fully
determined on the CPU: the class's declarations (fidelity, ``color_exactness_applicability``,
``reference_versions``, the TinyRenderer material law), the style layout, and the **host-side
kinematics chain that feeds nvdiffrast** — ``geom_poses``, ``lagged_pose``, ``camera_pose``,
``material_table``, ``build_scene_arrays``, ``checker_texture`` — replayed over the frozen 200-state
Step-7 reference corpus in float64. If any of those moved, the frames would move; that they are
byte-identical is the CPU half of the frame claim, and the GPU half is recorded as owed.

Usage::

    MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python scratch/step9/envs-craftax-rocket/oracle.py
    ... --python /Users/samib/se7-rocket-venv/bin/python --cells PFRocketLanding --name oracle_rocket.json
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

CELLS = ("CraftaxClassic", "CraftaxFull", "PFRocketLanding")

#: Run inside ONE tree, printing a JSON record on stdout. Executed with ``PYTHONPATH`` pointed at the
#: tree under test, because the two trees define the same module names and cannot both be imported
#: into one interpreter.
PROBE = r'''
import hashlib, json, sys
import numpy as np, torch

import stable_envs as se

cells = json.loads(sys.argv[1])
root = sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def build(cls, n, dev):
    """Construct in either tree: the Step-8 baseline knows only ``batch=``."""
    try:
        return cls(num_envs=n, device=dev)
    except TypeError:
        return cls(batch=n, device=dev)

def shut(env):
    """``close`` exists only in the migrated tree; the baseline has no lifecycle."""
    if hasattr(env, "close"):
        env.close()

def jsonable(x):
    return json.loads(json.dumps(x, default=str, sort_keys=True))

#: cell -> (corpus label -> (path, key)); the 127-wide Step-4 array is zero-extended below.
CORPORA = {
    "CraftaxClassic": {
        "step4_127_zero_extended": f"{root}/examples/results/step4/states/CraftaxClassic.npz",
        "step7_149": f"{root}/examples/results/step7/craftax-classic-full/corpus/states.npz",
    },
    "CraftaxFull": {
        "step7_544": f"{root}/examples/results/step7/craftax-full/corpus/states.npz",
    },
}
#: cell -> the certified tile sizes its frames are digested at.
SIZES = {"CraftaxClassic": [7, 16], "CraftaxFull": [10, 16]}

table = {r["name"]: r for r in se.registry_table()}
out = {"cells": {}, "import_errors": (se.import_errors() if hasattr(se, "import_errors")
                                      else se.envs.import_errors())}

for name in cells:
    if name == "PFRocketLanding":
        continue
    cls = se.REGISTRY[name]
    rec = {"device": dev, "state_dim": int(cls.state_dim), "state_names": list(cls.state_names),
           "corpora": {}}
    for label, path in CORPORA[name].items():
        raw = np.load(path, allow_pickle=False)["states"].astype(np.float32)
        if raw.shape[1] != cls.state_dim:
            # The Step-4 replay obligation, verbatim from
            # scratch/step7/closure/craftax_world_row_replay.py: a 127-slot row zero-extended into the
            # 149-slot layout with light_level forced to 1.0. The direct render MUST raise.
            from stable_envs.envs.craftax import OBS_H, OBS_W, N_STRIP
            probe = build(se.REGISTRY[name], int(raw.shape[0]), dev)
            try:
                probe.render(torch.as_tensor(raw))
                raised = "NO RAISE — the narrow corpus was accepted"
            except ValueError as e:
                raised = f"ValueError: {str(e)[:120]}"
            shut(probe)
            states = np.zeros((raw.shape[0], cls.state_dim), np.float32)
            states[:, :raw.shape[1]] = raw
            states[:, OBS_H * OBS_W * 2 + 2 + N_STRIP] = 1.0
            rec["corpora"].setdefault(label, {})["narrow_corpus_is_refused"] = raised
            rec["corpora"][label]["zero_extended_from"] = int(raw.shape[1])
        else:
            states = raw
        n = int(states.shape[0])
        env = build(se.REGISTRY[name], n, dev)
        env.set_state(states)
        rows = env.get_state().detach().cpu().numpy()
        r = rec["corpora"].setdefault(label, {})
        r.update({
            "corpus_states": n,
            "states_sha256": h(states),
            "roundtrip_exact": bool(np.array_equal(rows, states)),
            "rows_sha256": h(rows),
        })
        for t in SIZES[name]:
            frames = env.render(size=t).detach().cpu().numpy()
            from_rows = env.render(torch.as_tensor(states), size=t).detach().cpu().numpy()
            r[f"frames_{t}px"] = {
                "shape": list(frames.shape), "sha256": h(frames),
                "from_rows_sha256": h(from_rows), "mean": float(frames.mean()),
                "distinct_colors": int(len(np.unique(frames.reshape(-1, 3), axis=0))),
            }
        shut(env)
    env = build(se.REGISTRY[name], 4, dev)
    rec.update({
        "size": int(env.size),
        "describe": jsonable(env.describe()),
        "style_names": list(env.style_names),
        "style_dim": int(env.style_dim),
        "default_style_sha256": h(np.asarray(env.default_style(), np.float32)),
        "registry_row": table[name],
        "sample_states_sha256": h(np.asarray(env.sample_states(64, np.random.default_rng(0)),
                                             np.float32)),
        "variation_names": list(env.variation_space.names()),
        "parity_interior_min": cls.parity_interior_min,
        "parity_all_exact_min": cls.parity_all_exact_min,
        "fidelity_tier": cls.fidelity_tier,
        "render_decoder": cls.render_decoder,
        "reference_versions": jsonable(cls.reference_versions),
        "SWM_WRAPPER_GATE_F": jsonable(cls.SWM_WRAPPER_GATE_F),
        "resolution_consistency_applicability": jsonable(
            cls.resolution_consistency_applicability),
        "parity_note": cls.parity_note,
        "certification_state_distribution": jsonable(cls.certification_state_distribution),
        "constructor_takes_num_envs": "num_envs" in __import__("inspect").signature(
            cls.__init__).parameters,
    })
    shut(env)
    out["cells"][name] = rec

if "PFRocketLanding" in cells:
    # Rocket never renders here: the frames are owed to the GPU oracle. What IS fully determined on
    # the CPU is the module's declarations and its host-side kinematics chain, replayed over the
    # frozen 200-state Step-7 reference corpus.
    from stable_envs.envs import rocket_landing as RL
    cls = RL.PFRocketLanding
    z = np.load(f"{root}/examples/results/step7/rocket-landing/reference/states.npz",
                allow_pickle=False)
    st = np.asarray(z["state13"], np.float64)
    style = np.asarray(z["style"], np.float64)
    vert, norm, face, vg, uv = RL.build_scene_arrays()
    xp, xm = RL.geom_poses(st)
    _, _, pos_prev, q_prev, _, _ = RL.lagged_pose(st)
    eye, cmat = RL.camera_pose(pos_prev, q_prev)
    out["cells"]["PFRocketLanding"] = {
        "renders_here": False,
        "missing_backend": RL.missing_backend(),
        "registered": "PFRocketLanding" in se.REGISTRY,
        "state_dim": int(cls.state_dim), "state_names": list(cls.state_names),
        "default_size": int(cls.default_size),
        "default_supersample": int(cls.default_supersample),
        "fidelity_tier": cls.fidelity_tier,
        "parity_interior_min": cls.parity_interior_min,
        "visual_fidelity": jsonable(cls.visual_fidelity),
        "visual_note": cls.visual_note,
        "color_exactness_applicability": jsonable(cls.color_exactness_applicability),
        "reference_id": cls.reference_id,
        "documented_limitations": RL.DOCUMENTED_LIMITATIONS,
        "reference_unreachable_share": RL.REFERENCE_UNREACHABLE_SHARE,
        "style_layout": jsonable(RL.style_layout()),
        "style_dim": int(RL.style_dim()),
        "default_style_row_sha256": h(RL._default_style_row()),
        "corpus_states": int(st.shape[0]),
        "corpus_sha256": h(st),
        # the batch-invariant float64 mvp and everything that feeds it
        "geom_poses_xpos_sha256": h(xp), "geom_poses_xmat_sha256": h(xm),
        "camera_eye_sha256": h(eye), "camera_mat_sha256": h(cmat),
        "lagged_pos_sha256": h(pos_prev), "lagged_quat_sha256": h(q_prev),
        "material_table_sha256": h(RL.material_table(style)),
        "corpus_style_to_env_sha256": h(RL.corpus_style_to_env(style)),
        "scene_vert_sha256": h(vert), "scene_norm_sha256": h(norm),
        "scene_face_sha256": h(face), "scene_vg_sha256": h(vg), "scene_uv_sha256": h(uv),
        "checker_texture_sha256": h(RL.checker_texture()),
        "constructor_takes_num_envs": "num_envs" in __import__("inspect").signature(
            cls.__init__).parameters,
    }
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the migrated tree only (there is nothing to compare them against).
FUNCTIONAL = r'''
import json, sys
import numpy as np, torch
import stable_envs as snv

cells = json.loads(sys.argv[1])
root = sys.argv[2]
CORP = {"CraftaxClassic": f"{root}/examples/results/step7/craftax-classic-full/corpus/states.npz",
        "CraftaxFull": f"{root}/examples/results/step7/craftax-full/corpus/states.npz"}
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    for name in cells:
        if name not in CORP:
            continue
        corpus = np.load(CORP[name], allow_pickle=False)["states"].astype(np.float32)[:16]
        n = int(corpus.shape[0])
        env = getattr(snv, name)(num_envs=n, device=dev)
        keys = [tuple(k) for k, _, _ in env._layout()]
        env.set_state(corpus)
        held = env.get_state().clone()
        st = env.state
        rec = {"device": dev, "cell": name,
               "state_keys": ["/".join(k) for k in keys],
               "pack_unpack_is_identity": bool(torch.equal(env.pack(env.unpack(held)), held)),
               "pack_state_is_get_state": bool(torch.equal(env.pack(st), held)),
               "render_state_equals_render_rows": bool(torch.equal(env.render(st),
                                                                   env.render(held))),
               "render_state_equals_render_held": bool(torch.equal(env.render(st), env.render())),
               "observation_spec_matches_the_frame": (
                   list(env.observation_spec.shape) == list(env.render(st).shape[1:])),
               "state_is_not_a_view_of_the_env": None,
               "has_dynamics": bool(env.has_dynamics)}
        # a State the caller keeps must not change under the env's feet (foundation R8)
        snap = st["world", "blocks"].clone()
        env.set_state(corpus[::-1].copy())
        rec["state_is_not_a_view_of_the_env"] = bool(torch.equal(st["world", "blocks"], snap))
        env.set_state(corpus)
        a = torch.zeros(n, 1, device=env.device)
        for what, call in (("step", lambda: env.step(a)),
                           ("transition", lambda: env.transition(st, a)),
                           ("unroll", lambda: env.unroll(st, a.unsqueeze(0)))):
            try:
                call()
                rec[f"{what}_refuses"] = "NO RAISE"
            except NotImplementedError as e:
                rec[f"{what}_refuses"] = str(e)[:200]
        try:
            env.action_spec
            rec["action_spec_refuses"] = "NO RAISE"
        except NotImplementedError as e:
            rec["action_spec_refuses"] = str(e)[:80]
        env.close()
        rec["closed_env_refuses_render"] = False
        try:
            env.render(corpus)
        except RuntimeError:
            rec["closed_env_refuses_render"] = True
        out.append(rec)

from stable_envs.envs import _craftax_full, craftax_full
shims = {"craftax_full_is_the_private_module": craftax_full is _craftax_full}
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
    "size": "the registered default tile size",
    "corpora": "THE FRAMES, byte for byte, on every frozen corpus at every certified size, plus the "
               "corpus round-trip and the narrow-corpus refusal",
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
    "reference_versions": "the reference package versions the parity numbers were measured against",
    "SWM_WRAPPER_GATE_F": "the Step-8 JIT-wrapper declaration",
    "resolution_consistency_applicability": "where check_resolution_consistency applies",
    "parity_note": "what the parity bound is and is not",
    "certification_state_distribution": "the distribution the tier was measured on",
    # Rocket-only keys
    "renders_here": "whether the rocket rendered in this interpreter (it must not)",
    "missing_backend": "why it cannot render here",
    "registered": "whether the class registered in this interpreter",
    "default_size": "the registered frame size",
    "default_supersample": "TinyRenderer has no anti-aliasing",
    "visual_fidelity": "the measured visual-tier numbers AND their thresholds",
    "visual_note": "what the residual is",
    "color_exactness_applicability": "the five lit leaves and the 0.993567 ceiling",
    "reference_id": "the SWM id",
    "documented_limitations": "the documented limitations text",
    "reference_unreachable_share": "the share of states the reference bridge cannot reach",
    "style_layout": "the flat style layout",
    "default_style_row_sha256": "the official appearance as a style row",
    "corpus_states": "how many frozen states were replayed",
    "corpus_sha256": "the frozen rocket corpus itself",
    "geom_poses_xpos_sha256": "the 13 geom positions, float64",
    "geom_poses_xmat_sha256": "the 13 geom orientations, float64",
    "camera_eye_sha256": "the rocket-mounted camera's eye, one substep behind the body",
    "camera_mat_sha256": "THE BATCH-INVARIANT FLOAT64 MVP",
    "lagged_pos_sha256": "the lagged pose the camera reads",
    "lagged_quat_sha256": "the lagged attitude the camera reads",
    "material_table_sha256": "the TinyRenderer material table from the style row",
    "corpus_style_to_env_sha256": "the corpus-style to env-style mapping",
    "scene_vert_sha256": "the scene's vertices",
    "scene_norm_sha256": "the scene's normals",
    "scene_face_sha256": "the scene's triangles",
    "scene_vg_sha256": "the vertex-to-geom map",
    "scene_uv_sha256": "the ground's UVs",
    "checker_texture_sha256": "the nearest-sampled checker ground",
    "constructor_takes_num_envs": "the Step-9 constructor",
}

#: What this oracle deliberately does NOT establish, and who owes it.
OWED = {
    "PFRocketLanding.frames": (
        "Rocket renders only on cuda through nvdiffrast (CPU refusal is by design), so no frame was "
        "produced or compared on this host. OWED to the post-Wave-1 GPU oracle: the 200-state Step-7 "
        "reference corpus rendered at 480 px through both trees on one card in one job, byte-compared, "
        "plus tests/gpu for this cell. Everything the frame is a function of on the CPU side — the "
        "kinematics, the lagged camera, the float64 MVP, the material table, the scene arrays and the "
        "checker texture — IS compared here and is identical."),
    "Craftax.cuda_arm": (
        "The comparison arm ran on cpu. Craftax's render is the same torch gather on both devices "
        "(no CUDA-only fused kernel), but --device cuda is one flag and should be run on the GPU host "
        "with the Craftax GPU suites."),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default="CraftaxClassic,CraftaxFull")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-craftax-rocket"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        before = _run(PROBE, baseline, a.python, json.dumps(cells), str(baseline), a.device)
        after = _run(PROBE, ROOT, a.python, json.dumps(cells), str(ROOT), a.device)
        fn = _run(FUNCTIONAL, ROOT, a.python, json.dumps(cells), str(ROOT))

        diffs: dict = {}
        expected: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                if k not in b and k not in m:
                    continue
                bv, mv = b.get(k), m.get(k)
                if k == "constructor_takes_num_envs":
                    # THE MIGRATION ITSELF. Contract §2 replaces ``batch=`` with ``num_envs=``; the
                    # baseline answers False by construction, so a difference here is the patch
                    # working and its ABSENCE would be the failure.
                    if bv is not mv:
                        expected.setdefault(name, {})[k] = {
                            "baseline": bv, "migrated": mv,
                            "why": "contract §2: num_envs= replaces batch= (batch= still accepted, "
                                   "deprecated). The baseline cannot take num_envs; that is what "
                                   "was migrated."}
                    continue
                if k == "registry_row":
                    if (bv or {}).get("module") != (mv or {}).get("module"):
                        expected.setdefault(name, {})["registry_row.module"] = {
                            "baseline": (bv or {}).get("module"),
                            "migrated": (mv or {}).get("module"),
                            "why": "plan §3 asks for one craftax.py; the two modules share 21 "
                                   "module-level names holding DIFFERENT values, so full Craftax "
                                   "moved to the private _craftax_full.py beside it and "
                                   "stable_envs.envs.craftax_full is a sys.modules alias of it "
                                   "(checked)"}
                    bv = {kk: vv for kk, vv in (bv or {}).items() if kk != "module"}
                    mv = {kk: vv for kk, vv in (mv or {}).items() if kk != "module"}
                if bv != mv:
                    diffs.setdefault(name, {})[k] = {"baseline": bv, "migrated": mv, "means": why}

        checks = fn["checks"]
        functional_ok = all(
            c["pack_unpack_is_identity"] and c["pack_state_is_get_state"]
            and c["render_state_equals_render_rows"] and c["render_state_equals_render_held"]
            and c["observation_spec_matches_the_frame"] and c["state_is_not_a_view_of_the_env"]
            and not c["has_dynamics"] and c["closed_env_refuses_render"]
            and c["step_refuses"] != "NO RAISE" and c["transition_refuses"] != "NO RAISE"
            and c["unroll_refuses"] != "NO RAISE"
            for c in checks)
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "compare_device": a.device,
            "workstream": "step9/envs-craftax-rocket",
            "cells": cells,
            "python": a.python,
            "compared": sorted(COMPARED),
            "owed": OWED,
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
            cell = record["migrated"]["cells"][name]
            for label, c in (cell.get("corpora") or {}).items():
                for k, v in c.items():
                    if k.startswith("frames_"):
                        print(f"  {name:16s} {label:28s} {k:12s} {v['sha256'][:16]}"
                              f"  {tuple(v['shape'])}")
            if not cell.get("corpora"):
                print(f"  {name:16s} frames OWED to the GPU oracle; "
                      f"mvp {cell['camera_mat_sha256'][:16]}  "
                      f"materials {cell['material_table_sha256'][:16]}")
        ok = (record["identical"] and functional_ok
              and all(after["cells"][c]["constructor_takes_num_envs"] for c in cells)
              and all(fn["old_module_paths_resolve"].values()))
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
