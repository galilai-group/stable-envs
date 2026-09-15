"""M6: the Step-5 representative surfaces are untouched by the batch-invariant mvp.

Sami's re-freeze ruling asks this to be stated explicitly and PROVED, not asserted: the two surfaces
Step 5 rests on -- ``PushT`` with LeWM (``swm/PushT-v1``) and the DM Control Reacher with DINO-WM
(``swm/ReacherDMControl-v0`` -> ``DMCReacherEasy`` / ``DMCReacherHard``) -- do not render through
``stable_envs/render/nvdiff.py`` and their frozen Step-4 digests do not move.

Two independent proofs, because either alone is weak:

  1. **Structural, and checked by CALL rather than by import.** The frozen shards record each
     config's backends: ``PushT`` is ``["scene"]`` (``stable_envs/render/scene.py``, the raster2d
     path) and both Reachers are ``["mjwarp", "raycast3d"]`` (``stable_envs/render/raycast3d.py``).
     Neither lists nvdiff. A ``sys.modules`` check would be too weak to settle it --
     ``mujoco_locomotion.py`` imports ``..render.nvdiff`` at module scope, so the module is present
     in any interpreter where the six cells register, whatever else is rendering. So this script
     WRAPS the two entry points the change is in (``nvdiff.Scene.clip``, which holds the mvp, and
     ``nvdiff.BatchRenderer.render``) with a counter, renders the config, and reports the count.
     Zero calls is the claim "this frame did not go through the changed code", in a form that cannot
     be stale and does not depend on which optional packages the venv has.
  2. **Empirical.** It replays the config against its FROZEN Step-4 corpus
     (``examples/results/step4/states/<config>.npz``, loaded and never redrawn) through
     ``examples/fidelity_step4.py``'s OWN ``reseed`` and ``oracle`` -- the same functions the frozen
     numbers were produced by, imported rather than reimplemented -- and compares ``digest_ours``
     against the value in ``examples/results/step4/conformance/<shard>.json``.

Run it twice in one job, ``PYTHONPATH=$BASE`` and ``PYTHONPATH=$WS``, so the decisive comparison is
branch-vs-baseline on the same node in the same interpreter. The frozen-digest comparison is reported
beside it: if a digest fails to reproduce for a reason unrelated to this change (a different mujoco or
dm_control pin from the one Step 4 was frozen under), the BASELINE leg shows the same thing, and
branch == baseline still proves this change is not involved. Writes ``step5_surfaces_<tag>.json``;
``--phase compare`` prints the table and the verdict.

Usage:
    PYTHONPATH=$BASE python step5_surfaces.py --phase render --tag baseline --out DIR
    PYTHONPATH=$WS   python step5_surfaces.py --phase render --tag branch   --out DIR
    PYTHONPATH=$WS   python step5_surfaces.py --phase compare               --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

#: config id -> the frozen shard that records it
SURFACES = {"PushT": "raster", "DMCReacherEasy": "dmc", "DMCReacherHard": "dmc"}


def _code_root() -> Path:
    """The checkout the MEASURED ``stable_envs`` came from -- not the one this script lives in.

    The baseline leg runs this same file with ``PYTHONPATH=$SCR/se7/baseline``, so ``__file__`` points
    at the branch while the package under test is the baseline. Both the frozen Step-4 artifacts and
    ``fidelity_step4.py`` are therefore resolved from the package, and nothing from the branch is put
    on ``sys.path`` ahead of it.
    """
    import stable_envs as se
    return Path(os.path.dirname(os.path.dirname(os.path.realpath(se.__file__))))


def _frozen(root: Path, config_id: str) -> dict:
    shard = json.load(open(root / "examples" / "results" / "step4" / "conformance"
                           / f"{SURFACES[config_id]}.json"))
    return shard["configs"][config_id]


def phase_render(out: str, tag: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import stable_envs as se
    ROOT = _code_root()
    STEP4 = ROOT / "examples" / "results" / "step4"
    sys.path.insert(0, str(ROOT / "examples"))       # fidelity_step4 resolves the rest from its own file
    import fidelity_step4 as F                       # its own reseed/corpus_for/oracle, not a copy

    R = {"provenance": {"tag": tag, "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
                        "slurm_job": os.environ.get("SLURM_JOB_ID"),
                        "threads": torch.get_num_threads(),
                        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                        "pythonpath": os.environ.get("PYTHONPATH"),
                        "stable_envs_file": os.path.realpath(se.__file__),
                        "sha": (open(_code_root() / ".git_sha").read().strip()
                                if (_code_root() / ".git_sha").exists() else "unknown")},
         "configs": {}}
    for cid in SURFACES:
        row: dict = {}
        if cid not in se.REGISTRY:
            row["blocked"] = se.envs.import_errors()
            R["configs"][cid] = row
            print(tag, cid, "NOT REGISTERED", flush=True)
            continue
        frozen = _frozen(ROOT, cid)
        row["frozen_backends"] = frozen.get("backends")
        row["frozen_digest_ours"] = frozen["oracle"].get("digest_ours")
        row["frozen_reference_id"] = frozen["signature"]["describe"].get("reference_id")
        cls = se.REGISTRY[cid]
        env = se.make(cid, batch=64, device="cuda")
        states, meta = F.corpus_for(env, cid, STEP4 / "states", F.N_STATES, F.POOL)
        row["corpus"] = {"sha256": meta.get("sha256"), "n": int(len(states)),
                         "loaded_from": meta.get("loaded_from")}
        F.reseed(env)
        style = None
        row["oracle"] = F.oracle(env, {"config_id": cid}, states, style)
        row["digest_matches_frozen"] = bool(row["oracle"]["digest_ours"] == row["frozen_digest_ours"])
        row["renderer_module"] = type(env).__module__
        row["nvdiff_on_path"] = {
            "stable_envs.render.nvdiff_imported": "stable_envs.render.nvdiff" in sys.modules,
            "nvdiffrast_imported": "nvdiffrast" in sys.modules,
            **_nvdiff_call_count(env, states)}
        R["configs"][cid] = row
        print(tag, cid, json.dumps({"digest_ours": row["oracle"]["digest_ours"][:16],
                                    "matches_frozen": row["digest_matches_frozen"],
                                    "nvdiff_calls": row["nvdiff_on_path"].get("scene_clip_calls")}), flush=True)
        del env
        torch.cuda.empty_cache()
    _dump(os.path.join(out, f"step5_surfaces_{tag}.json"), R)


def _nvdiff_call_count(env, states) -> dict:
    """Render ``states`` with ``nvdiff.Scene.clip`` and ``nvdiff.BatchRenderer.render`` counted.

    The changed code is ``Scene.clip``'s mvp. If this config's frame never enters it, the change
    cannot reach this config -- which is the whole claim, made by execution instead of by reading.
    """
    import importlib
    try:
        ND = importlib.import_module("stable_envs.render.nvdiff")
    except Exception as e:                                                          # pragma: no cover
        return {"nvdiff_importable": False, "why": f"{type(e).__name__}: {e}",
                "scene_clip_calls": 0, "batch_render_calls": 0}
    counts = {"scene_clip_calls": 0, "batch_render_calls": 0}
    orig_clip, orig_render = ND.Scene.clip, ND.BatchRenderer.render

    def clip(self, *a, **k):
        counts["scene_clip_calls"] += 1
        return orig_clip(self, *a, **k)

    def render(self, *a, **k):
        counts["batch_render_calls"] += 1
        return orig_render(self, *a, **k)

    ND.Scene.clip, ND.BatchRenderer.render = clip, render
    try:
        env.render(states).cpu().numpy()
    finally:
        ND.Scene.clip, ND.BatchRenderer.render = orig_clip, orig_render
    return {"nvdiff_importable": True, **counts}


def phase_compare(out: str) -> None:
    b = json.load(open(os.path.join(out, "step5_surfaces_baseline.json")))
    n = json.load(open(os.path.join(out, "step5_surfaces_branch.json")))
    R = {"baseline_provenance": b["provenance"], "branch_provenance": n["provenance"], "configs": {}}
    ok = True
    for cid in SURFACES:
        cb, cn = b["configs"].get(cid, {}), n["configs"].get(cid, {})
        if "oracle" not in cb or "oracle" not in cn:
            R["configs"][cid] = {"blocked": {"baseline": cb.get("blocked"), "branch": cn.get("blocked")}}
            ok = False
            continue
        row = {"backends": cn.get("frozen_backends"),
               "renderer_module": cn.get("renderer_module"),
               "nvdiff_calls_baseline": {k: cb["nvdiff_on_path"][k] for k in
                                         ("scene_clip_calls", "batch_render_calls")},
               "nvdiff_calls_branch": {k: cn["nvdiff_on_path"][k] for k in
                                       ("scene_clip_calls", "batch_render_calls")},
               "nvdiff_module_imported_branch": cn["nvdiff_on_path"].get("stable_envs.render.nvdiff_imported"),
               "corpus_sha256": cn["corpus"]["sha256"],
               "frozen_digest_ours": cn["frozen_digest_ours"],
               "digest_ours_baseline": cb["oracle"]["digest_ours"],
               "digest_ours_branch": cn["oracle"]["digest_ours"],
               "branch_equals_baseline": cb["oracle"]["digest_ours"] == cn["oracle"]["digest_ours"],
               "branch_equals_frozen": cn["digest_matches_frozen"],
               "baseline_equals_frozen": cb["digest_matches_frozen"],
               "digest_official_baseline": cb["oracle"].get("digest_official"),
               "digest_official_branch": cn["oracle"].get("digest_official"),
               "mean_pixel_baseline": cb["oracle"].get("mean_pixel"),
               "mean_pixel_branch": cn["oracle"].get("mean_pixel")}
        row["untouched"] = bool(row["branch_equals_baseline"]
                                and row["nvdiff_calls_branch"]["scene_clip_calls"] == 0
                                and row["nvdiff_calls_branch"]["batch_render_calls"] == 0)
        ok &= row["untouched"]
        R["configs"][cid] = row
        print("compare", cid, "untouched" if row["untouched"] else "CHANGED",
              json.dumps({k: row[k] for k in ("branch_equals_baseline", "branch_equals_frozen",
                                              "nvdiff_calls_branch")}), flush=True)
    R["all_untouched"] = bool(ok)
    _dump(os.path.join(out, "step5_surfaces_compare.json"), R)
    print("STEP5_SURFACES_UNTOUCHED", R["all_untouched"], flush=True)


def _dump(path: str, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("render", "compare"))
    ap.add_argument("--tag", default="branch", choices=("baseline", "branch"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.phase == "render":
        phase_render(a.out, a.tag)
    else:
        phase_compare(a.out)


if __name__ == "__main__":
    main()
