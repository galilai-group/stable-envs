"""Oracle proof for the Workstream-C migration: the same Fetch, from two different trees.

WHAT IT PROVES, AND — because this family cannot make a pixel on this host — WHAT IT DOES NOT.

The Step-8 behavioral oracle (plan §7) says the architecture may change and the behavior may not.
This is ``scratch/step9/foundation/pusht_oracle.py``'s argument over the twelve Fetch cells, run
through BOTH trees in two separate interpreters — a detached ``git worktree`` of the Step-8 baseline
SHA and this branch, the SAME venv — and compared key by key.

THE FRAME DIFF IS OWED, AND IT IS OWED FOR A REASON THAT IS A PROPERTY OF THE FAMILY.
Fetch draws its 18 robot meshes with ``mujoco_warp``'s batch raytracer, which needs CUDA. There is
no CPU backend (``CPU_BACKEND_MISSING``, Step 3) and Step 9 does not add one. So ``render()`` cannot
be executed on this Mac in either tree, and the byte-for-byte frame comparison every other Wave-1
workstream reports is EXPLICITLY OWED to the post-Wave-1 GPU oracle, which the coordinator runs.
Nothing here should be read as a frame claim.

WHAT CAN STILL BE COMPARED WITHOUT A GPU, and is:

  1. ``describe()`` — the dict a consumer reads to decide whether these frames are usable;
  2. the style layout: ``style_names``, ``style_dim``, and the OFFICIAL DEFAULT APPEARANCE, which is
     the thing Sami's ruling forbids moving;
  3. the variation taxonomy and ``VariationMap.classification()`` per swm id, leaf for leaf, with the
     Step-8 counts (33 mapped / 12 partial / 12 unsupported / 15 dead / 51 n/a) asserted as totals;
  4. the fidelity declarations: tier, decoder, ``parity_interior_min``, ``parity_all_exact_min``,
     ``parity_note`` (verbatim), ``visual_fidelity``, ``variation_fidelity``, ``reference_versions``;
  5. the registry row for each of the twelve cells, including ``module``, and the twelve
     ``swm/Fetch*-v3`` aliases;
  6. ``state_dim``/``state_names``, and ``set_state``/``get_state`` round-tripping the FROZEN STEP-4
     CORPUS (``examples/results/step4/states/Fetch*.npz``) exactly;
  7. ``sample_states`` at a fixed seed — the env's own draw over that corpus;
  8. **THE PLATE BAKE**, which is the 91% of the frame that is exact by construction and is the one
     half of the renderer that runs on the CPU: ``_bake_plate`` is MuJoCo's own raster and is
     compared by SHA-256 at the default appearance and at a non-default one, per scene. It is the
     closest thing to a frame this host can produce, and it covers the plate cache key, the pristine
     restore, the camera recovery and the per-leaf model mutations;
  9. the mover-mask composition: ``movers``, the static set and the ``_mover_lut`` table, which is
     what decides which pixels the ray-cast is allowed to overwrite;
 10. the recovered camera poses under the default and a varied appearance (``_camera_batch``).

TWO THINGS THIS SCRIPT DOES TO BOTH TREES IDENTICALLY, and why they are not cheating:

  * ``warp``/``mujoco_warp`` are STUBBED in ``sys.modules`` so the module's import-time dependency
    check passes. Neither is touched by anything this script calls — the whole ray-cast path is
    skipped — and the stub is installed in both arms, so any difference it caused would appear in
    both. Without it the module refuses to import at all on this host and nothing is comparable.
  * ``_state_pool`` is replaced by a loader for the FROZEN STEP-4 CORPUS. The shipped pool builds
    itself by driving the official ``gymnasium.make("Fetch*-v4")``, and that raises
    ``AssertionError`` from ``gymnasium_robotics/utils/mujoco_utils.py::set_joint_qpos`` under
    numpy 2.5.3 + mujoco 3.12 — the upstream defect Step 3 already recorded at
    ``scratch/swm_matrix.py``'s ``PROBE_LIMITATIONS["fetch_cpu"]``. The corpus is the frozen record
    of what that pool produced on a working interpreter, so this substitutes the DATA and not the
    code path, and it is substituted in both arms. ``sample_states``'s own indexing, seeding and
    dtype are still the shipped ones and are still compared.

Usage::

    MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python scratch/step9/envs-fetch/oracle.py
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

#: The twelve cells. Each has a frozen Step-4 corpus at ``examples/results/step4/states/<Cell>.npz``.
SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
CELLS = tuple(f"Fetch{s}{x}" for x in ("", "Dense", "Dict") for s in SCENES)

#: The Step-8 variation taxonomy for this family, as the plan's §12-C line states it. Asserted as a
#: total over the twelve maps in BOTH trees, so a re-labelled leaf fails here and not in review.
TAXONOMY = {"mapped": 33, "partial": 12, "unsupported": 12, "dead_upstream": 15, "not_applicable": 51}

PROBE = r'''
import hashlib, json, os, sys, types
import importlib.machinery as _im
import numpy as np, torch

# The stub: see the module docstring. Installed before stable_envs is imported, in both arms.
for _nm in ("warp", "mujoco_warp"):
    if _nm not in sys.modules:
        _m = types.ModuleType(_nm)
        _m.__spec__ = _im.ModuleSpec(_nm, None)
        sys.modules[_nm] = _m
os.environ.setdefault("MUJOCO_GL", "glfw")

import stable_envs as se
import stable_envs.envs.fetch as F

cells = json.loads(sys.argv[1])
states_dir = sys.argv[2]

_POOLS = {}
def _pool(scene, ref_id, state_dim, nq, n=512):
    """The frozen Step-4 corpus, standing in for the live-env pool. See the module docstring."""
    if scene not in _POOLS:
        _POOLS[scene] = np.load(f"{states_dir}/Fetch{scene}.npz")["states"].astype(np.float32)
    return _POOLS[scene]
F._state_pool = _pool

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def jsonable(x):
    return json.loads(json.dumps(x, default=str))

try:
    from stable_envs.compat.variation_maps import fetch as FM
except Exception:
    from stable_envs._interop.variation_maps import fetch as FM

table = {r["name"]: r for r in se.registry_table()}
out = {"cells": {}, "maps": {}, "taxonomy": {}}

for swm_id, vm in sorted(FM.MAPS.items()):
    cl = vm.classification()
    out["maps"][swm_id] = jsonable(cl)
    for leaf, label in cl.items():
        out["taxonomy"][label] = out["taxonomy"].get(label, 0) + 1

for name in cells:
    corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)
    n = int(corpus.shape[0])
    cls = se.REGISTRY[name]
    try:
        env = cls(num_envs=n, device="cpu")
    except TypeError:
        env = cls(batch=n, device="cpu")
    env.set_state(corpus)
    rows = env.get_state().detach().cpu().numpy()

    # The plate: MuJoCo's own raster of the static scene, the 91% of the frame that is exact by
    # construction. Baked at the default appearance and at a deliberately non-default one that
    # touches EVERY leaf in PLATE_PATHS, so a changed key, a lost restore or a moved camera shows up.
    varied = {"floor.color": np.array([0.2, 0.4, 0.9]), "table.color": np.array([0.7, 0.1, 0.1]),
              "light.diffuse": np.array([1.3]), "camera.distance_scale": np.array([1.2]),
              "camera.azimuth_delta": np.array([11.0]), "camera.elevation_delta": np.array([-7.0])}
    plate_default = env._plate(64, None).detach().cpu().numpy()
    plate_varied = env._plate(64, varied).detach().cpu().numpy()

    # The camera, recovered from MuJoCo's own mjvScene, for the whole batch under the default style.
    sty = env.resolve_style(None, None, batch=n)
    campos, camquat = env._camera_batch(sty, n)

    # THE GROUPING DECISION, which is the one piece of render() that the migration rewrote and that
    # a CPU host can still execute: how a batch of DIFFERENT per-environment appearances is cut into
    # distinct plate bakes. The migrated tree answers with _style_groups; the baseline has the same
    # five lines written out inside render() and _camera_batch, so they are inlined here for it. The
    # comparison is therefore between the two IMPLEMENTATIONS, over the same sampled style.
    rsty = env.sample_style(batch=n, paths=list(env.PLATE_PATHS),
                            rng=np.random.default_rng(7))
    rsty = env.resolve_style(rsty, None, batch=n)
    groups = getattr(F, "_style_groups", None)
    if groups is not None:
        g_uniq, g_inv, g_offs = groups(rsty, env.PLATE_PATHS, n)
    else:
        g_offs, _o = [], 0
        for _p in env.PLATE_PATHS:
            _w = int(rsty.layout[_p][1])
            g_offs.append((_o, _w))
            _o += _w
        _key = torch.cat([rsty[_p].reshape(n, -1) for _p in env.PLATE_PATHS], 1).cpu().numpy()
        g_uniq, g_inv = np.unique(np.round(_key.astype(np.float64), 6), axis=0,
                                  return_inverse=True)
    # and the plates those groups actually bake, which is the whole point of the grouping
    group_plates = [h(env._plate(48, {p: row[a:a + w] for p, (a, w) in zip(env.PLATE_PATHS, g_offs)})
                      .detach().cpu().numpy()) for row in g_uniq]

    out["cells"][name] = {
        "state_dim": int(env.state_dim),
        "state_names": list(env.state_names),
        "size": int(env.size),
        "supersample": int(env.ss),
        "scene": env.SCENE,
        "ref_id": env.REF_ID,
        "corpus_states": n,
        "roundtrip_exact": bool(np.array_equal(rows, corpus)),
        "rows_sha256": h(rows),
        "describe": jsonable(env.describe()),
        "style_names": list(env.style_names),
        "style_dim": int(env.style_dim),
        "default_style_sha256": h(np.asarray(env.default_style(), np.float64)),
        "default_style": np.asarray(env.default_style(), np.float64).round(9).tolist(),
        "registry_row": jsonable(table[name]),
        "sample_states_sha256": h(np.asarray(env.sample_states(16, np.random.default_rng(0)),
                                             np.float32)),
        "variation_names": list(env.variation_space.names()),
        "plate_paths": list(env.PLATE_PATHS),
        "plate_default_sha256": h(plate_default),
        "plate_varied_sha256": h(plate_varied),
        "plate_shape": list(plate_default.shape),
        "movers": np.asarray(env._b.movers, np.int64).tolist(),
        "static": list(map(int, env._b.static)),
        "mover_lut_sha256": h(env._mover_lut().detach().cpu().numpy()),
        "campos_sha256": h(campos.detach().cpu().numpy()),
        "style_groups_uniq_sha256": h(np.asarray(g_uniq, np.float64)),
        "style_groups_inv": np.asarray(g_inv, np.int64).tolist(),
        "style_groups_offsets": [list(map(int, t)) for t in g_offs],
        "style_group_plate_sha256": group_plates,
        "camquat_sha256": h(camquat.detach().cpu().numpy()),
        "cam_defaults": {k: (v.tolist() if hasattr(v, "tolist") else float(v))
                         for k, v in env._b.cam_defaults.items()},
        "geom_names": {str(k): v for k, v in env._b.info["names"].items()},
        "fidelity_tier": env.fidelity_tier,
        "render_decoder": env.render_decoder,
        "parity_interior_min": env.parity_interior_min,
        "parity_all_exact_min": env.parity_all_exact_min,
        "parity_note": env.parity_note,
        "visual_fidelity": jsonable(env.visual_fidelity),
        "variation_fidelity": jsonable(env.variation_fidelity),
        "reference_versions": jsonable(env.reference_versions),
        "default_size": int(type(env).default_size),
    }
    if hasattr(env, "close"):
        env.close()
print("@@JSON@@" + json.dumps(out))
'''

#: The Step-9 claims, checked in the MIGRATED tree only: there is nothing at Step 8 to compare them
#: against, because none of them existed.
FUNCTIONAL = r'''
import json, os, sys, types
import importlib.machinery as _im
import numpy as np, torch
for _nm in ("warp", "mujoco_warp"):
    if _nm not in sys.modules:
        _m = types.ModuleType(_nm); _m.__spec__ = _im.ModuleSpec(_nm, None); sys.modules[_nm] = _m
os.environ.setdefault("MUJOCO_GL", "glfw")

import stable_envs as snv
import stable_envs.envs.fetch as F
from stable_envs import dynamics as D
from stable_envs.registry import SWM_IDS

cells = json.loads(sys.argv[1])
states_dir = sys.argv[2]
_POOLS = {}
def _pool(scene, ref_id, state_dim, nq, n=512):
    if scene not in _POOLS:
        _POOLS[scene] = np.load(f"{states_dir}/Fetch{scene}.npz")["states"].astype(np.float32)
    return _POOLS[scene]
F._state_pool = _pool

checks = []
for name in cells:
    cls = getattr(snv, name)
    corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)[:8]
    env = cls(num_envs=8, device="cpu")
    env.set_state(corpus)
    st = env.state
    keys = [".".join(k) if isinstance(k, tuple) else k for k in st.keys(True, True)]
    rec = {
        "cell": name,
        "num_envs_kwarg": int(env.num_envs) == 8 and len(env) == 8,
        "state_keys": keys,
        "pack_is_the_packed_row": bool(torch.equal(env.pack(st), env.get_state())),
        "unpack_pack_roundtrip": bool(torch.equal(
            env.pack(env.unpack(env.get_state())), env.get_state())),
        "nothing_hidden": all(o is not None for _, _, o in env._layout()),
        "has_dynamics": bool(env.has_dynamics),
        "swm_ids": [k for k, v in SWM_IDS.items() if name in v],
    }
    for what in ("step", "transition"):
        try:
            getattr(env, what)(st, None) if what == "transition" else env.step(None)
            rec[what + "_refusal"] = None
        except NotImplementedError as e:
            rec[what + "_refusal"] = str(e)
    # close(): idempotent, and a closed env refuses to render rather than drawing through a
    # context the caller believes is gone.
    env.close(); env.close()
    rec["close_idempotent"] = bool(env.closed)
    try:
        env.render()
        rec["closed_env_refuses_render"] = False
    except RuntimeError:
        rec["closed_env_refuses_render"] = True
    # batch= still works and means the same thing
    env2 = cls(batch=3, device="cpu")
    rec["batch_kwarg_still_works"] = int(env2.num_envs) == 3
    env2.close()
    checks.append(rec)
print("@@JSON@@" + json.dumps({"checks": checks}))
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
    "state_dim": "the packed row's width",
    "state_names": "the packed row's column names",
    "size": "the registered default frame size",
    "supersample": "the registered default supersample",
    "scene": "which of the four scenes this cell renders",
    "ref_id": "the official gymnasium-robotics id it is validated against",
    "corpus_states": "how many frozen states were replayed",
    "roundtrip_exact": "set_state/get_state round-tripping the frozen corpus",
    "rows_sha256": "the packed rows the renderer would be handed",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "THE OFFICIAL DEFAULT APPEARANCE (the one Sami's ruling forbids moving)",
    "default_style": "the official default appearance, value by value",
    "registry_row": "the registry row, module included",
    "sample_states_sha256": "the env's own draw at a fixed seed",
    "variation_names": "the variation taxonomy, leaf for leaf",
    "plate_paths": "which leaves re-bake the plate",
    "plate_default_sha256": "THE BAKED PLATE at the default appearance — 91% of the frame",
    "plate_varied_sha256": "THE BAKED PLATE under a non-default appearance touching every plate leaf",
    "plate_shape": "the plate's shape",
    "movers": "which geoms the ray-cast owns",
    "static": "which geoms the plate owns",
    "mover_lut_sha256": "the mover mask the composite indexes with",
    "campos_sha256": "the recovered camera position, for the whole batch",
    "style_groups_uniq_sha256": "the DISTINCT plate appearances a mixed batch is cut into",
    "style_groups_inv": "which plate each environment of a mixed batch is composited against",
    "style_groups_offsets": "where each plate leaf sits inside a grouped row",
    "style_group_plate_sha256": "the plates a mixed batch actually bakes, one per group",
    "camquat_sha256": "the recovered camera orientation, for the whole batch",
    "cam_defaults": "the official free-camera configuration se:cam was built from",
    "geom_names": "the geom names the surgical model compiles to",
    "fidelity_tier": "the fidelity tier",
    "render_decoder": "the decoder declaration",
    "parity_interior_min": "the declared interior-parity floor",
    "parity_all_exact_min": "the declared all-pixel floor",
    "parity_note": "the declared residual, verbatim",
    "visual_fidelity": "the visual-fidelity declaration",
    "variation_fidelity": "the variation-fidelity declaration",
    "reference_versions": "the reference-package provenance",
    "default_size": "the class's registered default size",
}

#: What this host cannot execute, stated as a debt rather than left as a silence.
OWED = {
    "frames": (
        "render() for all twelve cells, byte for byte, at the registered default 480 px over the "
        "frozen Step-4 corpora, and the GPU timings. Fetch's mover pass is mujoco_warp's batch "
        "raytracer: CUDA-only, no CPU backend (CPU_BACKEND_MISSING, Step 3), so neither tree can "
        "make a Fetch pixel on this Mac. OWED TO THE POST-WAVE-1 GPU ORACLE, which the coordinator "
        "runs; this script takes --python and needs no other change there."),
    "live_reference": (
        "reference_env()/reference_render() and the shipped _state_pool, all of which build the "
        "official gymnasium.make('Fetch*-v4'). UNMEASURED-locally: that raises AssertionError from "
        "gymnasium_robotics/utils/mujoco_utils.py::set_joint_qpos under numpy 2.5.3 + mujoco 3.12, "
        "the upstream defect recorded at scratch/swm_matrix.py::PROBE_LIMITATIONS['fetch_cpu']. "
        "Neither was touched by this migration."),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--head-sha", default=None)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/envs-fetch"))
    ap.add_argument("--name", default="oracle.json")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        before = _run(PROBE, baseline, a.python, json.dumps(cells),
                      str(baseline / "examples/results/step4/states"))
        after = _run(PROBE, ROOT, a.python, json.dumps(cells),
                     str(ROOT / "examples/results/step4/states"))
        fn = _run(FUNCTIONAL, ROOT, a.python, json.dumps(cells),
                  str(ROOT / "examples/results/step4/states"))

        diffs: dict = {}
        for name in cells:
            b, m = before["cells"][name], after["cells"][name]
            for k, why in COMPARED.items():
                if b[k] != m[k]:
                    diffs.setdefault(name, {})[k] = {"baseline": b[k], "migrated": m[k],
                                                     "means": why}
        if before["maps"] != after["maps"]:
            diffs["variation_maps"] = {"classification": {"baseline": before["maps"],
                                                          "migrated": after["maps"],
                                                          "means": "the Step-8 variation taxonomy"}}
        if before["taxonomy"] != after["taxonomy"]:
            diffs["variation_maps_totals"] = {"baseline": before["taxonomy"],
                                              "migrated": after["taxonomy"]}

        taxonomy_ok = after["taxonomy"] == TAXONOMY
        c = fn["checks"]
        swm_ok = sorted(i for r in c for i in r["swm_ids"]) == sorted(
            f"swm/{n}-v3" for n in cells)
        functional_ok = bool(c) and all(
            r["num_envs_kwarg"] and r["state_keys"] == ["qpos", "goal"]
            and r["pack_is_the_packed_row"] and r["unpack_pack_roundtrip"]
            and r["nothing_hidden"] and not r["has_dynamics"]
            and r["step_refusal"] and r["transition_refusal"]
            and r["close_idempotent"] and r["closed_env_refuses_render"]
            and r["batch_kwarg_still_works"] and len(r["swm_ids"]) == 1 for r in c)

        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": a.head_sha or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
            "workstream": "step9/envs-fetch",
            "compare_device": "cpu (the plate half only; see owed_comparisons)",
            "cells": cells,
            "corpus": "examples/results/step4/states/<Cell>.npz",
            "python": a.python,
            "compared": sorted(COMPARED),
            "owed_comparisons": OWED,
            "unexplained_differences": diffs,
            "expected_differences": {},
            "identical": not diffs,
            "taxonomy": after["taxonomy"],
            "taxonomy_expected": TAXONOMY,
            "taxonomy_ok": taxonomy_ok,
            "swm_aliases_ok": swm_ok,
            "functional_checks": c,
            "functional_ok": functional_ok,
            "baseline": before,
            "migrated": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / a.name).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

        print(json.dumps({k: record[k] for k in (
            "baseline_sha", "head_sha", "identical", "unexplained_differences", "taxonomy",
            "taxonomy_ok", "swm_aliases_ok", "functional_ok")}, indent=1))
        for name in cells:
            r = record["migrated"]["cells"][name]
            print(f"  {name:26s} plate {r['plate_default_sha256'][:16]}"
                  f"  rows {r['rows_sha256'][:12]}  states {r['corpus_states']:4d}")
        ok = record["identical"] and taxonomy_ok and swm_ok and functional_ok
        print("ORACLE:", "PASS" if ok else "FAIL", "(frames owed to the GPU oracle)")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
