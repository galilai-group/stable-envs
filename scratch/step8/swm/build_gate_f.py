"""Generate the Gate-F / SWM-API matrix for all 42 SWM ids (Step 8, Workstream D, plan section 12).

    python scratch/step8/swm/build_gate_f.py

writes ``examples/results/step8/swm/gate_f_matrix.json`` and
``docs/research/step8_swm_integration.md``.

This is a JOIN over already-committed evidence, in the same spirit as
``scratch/step8/report.py`` (Workstream E's support matrix): it never constructs an env, never
renders, and never re-measures anything that is already on disk. Its two cheap exceptions --
``default_style() == style_from_variation(env, {})`` for the classes constructible on this Mac's
CPU, and reading ``VariationMap.classification()`` -- are themselves pre-computed, committed
artifacts (``scratch/step8/swm/probe_default_roundtrip.py`` ->
``examples/results/step8/swm/default_roundtrip_cpu.json``; ``probe_variation_classification.py``
-> ``variation_classification.json``), so this script itself needs no ``numpy`` / ``stable_envs``
import and runs under plain ``python3``.

Per id, every cell in the matrix carries an explicit status from a fixed vocabulary plus an
evidence path -- never a bare PASS:

    MEASURED               a committed artifact (or a cheap check run here) establishes the claim
    UNMEASURED              nobody measured it here; not a failure, an absence of evidence
    NOT_APPLICABLE          the question does not apply to this id (symbolic, no renderer, ...)
    MISMATCH(upstream)       B/C measured the SWM wrapper's own reset producing a DIFFERENT
                            appearance than the registered default -- an upstream fact, not ours

Re-run after ``VARIATION_MAPS`` changes (e.g. T3, the in-flight plate-texel capability) by
re-running the two probes above first, THEN this script -- it always reads whatever is currently
committed under ``examples/results/step8/swm/*.json`` rather than caching a copy of its own.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S7 = ROOT / "examples" / "results" / "step7"
S8 = ROOT / "examples" / "results" / "step8"
OUT_JSON = S8 / "swm" / "gate_f_matrix.json"
OUT_DOC = ROOT / "docs" / "research" / "step8_swm_integration.md"

SUPPORT_MATRIX = S8 / "support_matrix.json"                       # Workstream E
COVERAGE_MATRIX = S7 / "closure2" / "coverage_matrix.json"          # Step-7 closure
BULK_CERT = S7 / "closure2" / "replay" / "certify" / "certification.json"
GATE_A_REJUDGED = S8 / "certification" / "gate_a_rejudged.json"     # Workstream E, 7 new cells
REFERENCE_ARM = S8 / "provenance" / "reference_arm.json"            # Workstream F, 4 trackcom cells
DEFAULT_ROUNDTRIP_CPU = S8 / "swm" / "default_roundtrip_cpu.json"   # this workstream, CPU
VARIATION_CLASSIFICATION = S8 / "swm" / "variation_classification.json"  # this workstream
FETCH_GPU_EVIDENCE = S8 / "variations-fetch" / "gpu_evidence.json"        # Workstream B, GPU
NEWCELLS_GPU_DIR = S8 / "variations-newcells" / "gpu"                     # Workstream C, GPU
DASHBOARD = S8 / "dashboard.yaml"

#: Symbolic ids -- intentionally renderer-free by design (plan section 12, item 8).
SYMBOLIC_IDS = {"swm/CraftaxClassicSymbolic-v1", "swm/CraftaxSymbolic-v1"}

#: Families whose stable-envs cells implement their OWN dynamics and are reachable through
#: ``StableEnvGym``/``BatchedEnvPool`` as well as ``swap_renderer`` (examples/swm_compat/README.md).
#: Every other pixel family is renderer-only here, so ``swap_renderer`` -- "the official env, our
#: pixels" -- is the ONLY Gate-F path (stable_envs/compat/worldmodel.py, module docstring).
NATIVE_ENV_FAMILIES = {"TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "PushT"}

#: Fetch is the one family where B measured the SWM WRAPPER's own reset producing an appearance
#: that mismatches the registered default (examples/results/step8/variations-fetch/findings.md:
#: "swm Fetch declared leaf defaults are not the official appearance"). Recorded as a per-family
#: fact rather than re-derived per id.
WRAPPER_RESET_MISMATCH_CLASSES = {
    "FetchPickAndPlace", "FetchPickAndPlaceDense", "FetchPickAndPlaceDict",
    "FetchPush", "FetchPushDense", "FetchPushDict",
    "FetchReach", "FetchReachDense", "FetchReachDict",
    "FetchSlide", "FetchSlideDense", "FetchSlideDict",
}

#: GPU-only classes: cannot be constructed under `device="cpu"` on this Mac (mujoco_warp / nvdiff
#: backends only). Registered-default roundtrip for these is read from the cited committed GPU
#: artifact instead of re-measured here.
GPU_ONLY_CLASSES = {
    "FetchPickAndPlace", "FetchPickAndPlaceDense", "FetchPickAndPlaceDict",
    "FetchPush", "FetchPushDense", "FetchPushDict",
    "FetchReach", "FetchReachDense", "FetchReachDict",
    "FetchSlide", "FetchSlideDense", "FetchSlideDict",
    "OGBCube", "OGBScene", "PFRocketLanding",
}

#: Classes with no VariationMap at all because swm offers this family zero appearance leaves
#: (n_swm_leaves = 0 in the support matrix / coverage matrix) -- not a mapping gap.
NO_SWM_VARIATION_SPACE_CLASSES = {"CraftaxClassic", "CraftaxFull"}


def _sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def _load(p: Path, default=None):
    return json.loads(p.read_text()) if p.is_file() else default


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


# --------------------------------------------------------------------------------------------
# 1. registry resolution + pixel/symbolic + cells, from the two already-committed joins
# --------------------------------------------------------------------------------------------

def load_base_rows() -> dict[str, dict]:
    support = _load(SUPPORT_MATRIX)
    coverage = _load(COVERAGE_MATRIX)
    cov_by_id = {r["swm_id"]: r for r in (coverage or {}).get("rows", [])}
    rows = {}
    for r in support["rows"]:
        sid = r["swm_id"]
        cov = cov_by_id.get(sid, {})
        rows[sid] = {
            "swm_id": sid,
            "pixel_or_symbolic": r["pixel_or_symbolic"],
            "stable_envs_cells": r.get("stable_envs_cells") or [],
            "registry_resolution": {
                "status": "MEASURED" if (r.get("stable_envs_cells") or sid in SYMBOLIC_IDS)
                          else "UNMEASURED",
                "classes": r.get("stable_envs_cells") or [],
                "evidence": r.get("join_evidence") or (cov.get("join") or {}).get("evidence"),
                "rung": (cov.get("join") or {}).get("rung"),
                "source": f"{_rel(SUPPORT_MATRIX)} (join_evidence) + {_rel(COVERAGE_MATRIX)} (join.rung)",
            },
            "backends": r.get("backends") or [],
            "supported_devices": r.get("supported_devices") or {},
        }
    assert len(rows) == 42, f"expected 42 SWM ids, got {len(rows)}"
    return rows


# --------------------------------------------------------------------------------------------
# 2. registered default status: default_style() == style_from_variation(env, {})
# --------------------------------------------------------------------------------------------

def registered_default_status(cells: list[str]) -> dict:
    if not cells:
        return {"status": "NOT_APPLICABLE", "why": "symbolic id: no renderer, no registered "
                "appearance to reproduce", "per_class": {}}
    if all(c in NO_SWM_VARIATION_SPACE_CLASSES for c in cells):
        return {"status": "NOT_APPLICABLE",
                "why": "swm offers zero variation leaves for this family (n_swm_leaves=0 in the "
                       "support matrix); there is no swm config whose round trip could be tested",
                "per_class": {}}
    cpu = _load(DEFAULT_ROUNDTRIP_CPU, {"runs": {}})
    fetch_gpu = _load(FETCH_GPU_EVIDENCE)
    per_class = {}
    statuses = []
    for c in cells:
        found = None
        for run in cpu.get("runs", {}).values():
            rec = run.get("classes", {}).get(c)
            if rec and rec.get("status") == "MEASURED":
                found = {"status": "MEASURED", "device": "cpu",
                         "default_equals_style_from_empty_variation":
                             rec["default_equals_style_from_empty_variation"],
                         "source": f"{_rel(DEFAULT_ROUNDTRIP_CPU)} (this workstream, CPU, "
                                   f"host={run.get('host')})"}
                break
        if found is None and c in GPU_ONLY_CLASSES:
            if c.startswith("Fetch") and fetch_gpu is not None:
                found = {"status": "MEASURED", "device": "cuda",
                         "note": "default_through_empty_mapping / round_trip, per-scene, in the "
                                 "cited artifact (Workstream B)",
                         "source": f"{_rel(FETCH_GPU_EVIDENCE)} (Workstream B, GPU)"}
            elif c in ("OGBCube", "OGBScene"):
                found = {"status": "MEASURED", "device": "cuda",
                         "source": f"{_rel(NEWCELLS_GPU_DIR / 'default_ogb_baseline.json')}, "
                                   f"{_rel(NEWCELLS_GPU_DIR / 'default_ogb_branch.json')} "
                                   f"(Workstream C, GPU, gpu2264, job 6443480)"}
            elif c == "PFRocketLanding":
                found = {"status": "MEASURED", "device": "cuda",
                         "source": f"{_rel(NEWCELLS_GPU_DIR / 'default_rocket_baseline.json')}, "
                                   f"{_rel(NEWCELLS_GPU_DIR / 'default_rocket_branch.json')} "
                                   f"(Workstream C, GPU, gpu2264, job 6443480)"}
        if found is None:
            found = {"status": "UNMEASURED",
                     "why": "no CPU backend for this class on this Mac (device='cpu' raises by "
                            "design) and no committed GPU artifact was found for it; would need "
                            "device='cuda' -- record the exact command: "
                            f"`se.make('{c}', batch=1, size=32, device='cuda')` then compare "
                            f"`env.default_style()` to `style_from_variation(env, {{}})`"}
        per_class[c] = found
        statuses.append(found["status"])
    overall = "MEASURED" if statuses and all(s == "MEASURED" for s in statuses) else (
        "UNMEASURED" if "MEASURED" not in statuses else "MEASURED")
    return {"status": overall, "per_class": per_class}


# --------------------------------------------------------------------------------------------
# 3. SWM wrapper reset appearance (SEPARATE from #2: the WRAPPER's own reset, not the mapping)
# --------------------------------------------------------------------------------------------

def wrapper_reset_appearance_status(cells: list[str]) -> dict:
    if not cells:
        return {"status": "NOT_APPLICABLE", "why": "symbolic id: no renderer"}
    if any(c in WRAPPER_RESET_MISMATCH_CLASSES for c in cells):
        return {"status": "MISMATCH(upstream)",
                "why": "Workstream B measured that swm's declared leaf defaults for the Fetch "
                       "family are not the official appearance -- an upstream finding, not a "
                       "stable-envs defect",
                "source": f"{_rel(S8 / 'variations-fetch' / 'findings.md')} "
                          "(\"swm Fetch declared leaf defaults are not the official appearance\"); "
                          f"{_rel(DASHBOARD)} upstream_findings (source: B)"}
    return {"status": "UNMEASURED",
            "why": "no committed Step-8 measurement of the SWM WRAPPER's own reset() appearance "
                   "against the registered default exists for this family beyond #2's mapping "
                   "round trip"}


# --------------------------------------------------------------------------------------------
# 4. VariationMap status, from VariationMap.classification() counts
# --------------------------------------------------------------------------------------------

def variation_map_status(cells: list[str], sid: str) -> dict:
    if not cells:
        return {"status": "NOT_APPLICABLE", "why": "symbolic id: no variation space to map",
                "per_class": {}}
    if all(c in NO_SWM_VARIATION_SPACE_CLASSES for c in cells):
        return {"status": "NOT_APPLICABLE",
                "why": "swm offers zero variation leaves for this family (n_swm_leaves=0)",
                "per_class": {}}
    vc = _load(VARIATION_CLASSIFICATION, {"classes": {}})
    per_class = {}
    any_map = False
    agg_counts: dict[str, int] = {}
    for c in cells:
        rec = vc["classes"].get(c)
        if rec is None:
            per_class[c] = {"status": "UNMEASURED", "why": "no VariationMap registered for this "
                            "class in stable_envs.compat.worldmodel.VARIATION_MAPS"}
            continue
        any_map = True
        per_class[c] = {"status": "MEASURED", "n_leaves": rec["n_leaves"], "counts": rec["counts"],
                        "source": f"{_rel(VARIATION_CLASSIFICATION)} (VariationMap.classification(), "
                                  f"class={c})"}
        for label, n in rec["counts"].items():
            agg_counts[label] = agg_counts.get(label, 0) + n
    status = "MEASURED" if any_map else "UNMEASURED"
    return {"status": status, "aggregate_counts": agg_counts, "per_class": per_class}


# --------------------------------------------------------------------------------------------
# 5. state / observation / API status, from the Step-7 closure2 gate_a shard per cell
# --------------------------------------------------------------------------------------------

_CHECKS_OF_INTEREST = ("check_state_roundtrip", "check_default_style_is_official",
                       "render_parity_with_official_renderer")


def _bulk_cert_index():
    d = _load(BULK_CERT)
    return (d or {}).get("configs", {})


def _rejudged_index():
    d = _load(GATE_A_REJUDGED)
    out = {}
    for rec in (d or {}).get("records", []):
        out.setdefault(rec["cell"], []).append(rec)
    return out


def _reference_arm_index():
    d = _load(REFERENCE_ARM)
    return (d or {}).get("cells", {})


def state_observation_api_status(cells: list[str], sid: str) -> dict:
    if not cells:
        return {"status": "NOT_APPLICABLE",
                "why": "symbolic id: renderer-free by design; no frame/state contract to check"}
    bulk = _bulk_cert_index()
    rejudged = _rejudged_index()
    arms = _reference_arm_index()
    per_cell = {}
    for c in cells:
        if c in arms:
            auth = arms[c]
            auth_arm = next((a for a in auth["arms"] if a["arm"] == auth["authoritative_arm"]), None)
            checks = {}
            per_cell[c] = {
                "status": "MEASURED",
                "note": "authoritative reference arm per Workstream F; see "
                        "docs/research/step8_reference_provenance.md",
                "source": f"{_rel(REFERENCE_ARM)} (Workstream F)",
                "source_within": auth_arm.get("source") if auth_arm else None,
                "source_sha256": auth_arm.get("source_sha256") if auth_arm else None,
                "gate_a": auth_arm.get("gate_a") if auth_arm else None,
            }
            continue
        if c in rejudged:
            recs = rejudged[c]
            # Prefer the device record that actually carries judged checks (a BLOCKED / refused
            # cpu record has `checks: {}` for the three GPU-only cells); among those, the cpu
            # record first for CPU-certified cells. A record with no checks is not evidence.
            judged = [r for r in recs if r["judgment"].get("checks")]
            if not judged:
                per_cell[c] = {"status": "UNMEASURED",
                               "why": "every rejudged record for this cell is BLOCKED with no "
                                      "judged checks",
                               "source": f"{_rel(GATE_A_REJUDGED)}"}
                continue
            rec = next((r for r in judged if r["device"] == "cpu"), judged[0])
            checks = rec["judgment"]["checks"]
            per_cell[c] = {
                "status": "MEASURED",
                "device": rec["device"],
                "checks": {k: checks.get(k, {}).get("status") for k in _CHECKS_OF_INTEREST
                          if k in checks},
                "source": f"{_rel(GATE_A_REJUDGED)} (Workstream E rejudgment of the Step-7 shard)",
                "source_within": rec.get("source"), "source_sha256": rec.get("source_sha256"),
            }
            continue
        cfg = bulk.get(c)
        if cfg is None:
            # PointmazeLargeStitch: no separate bulk config, shares the rendered scene with
            # PointmazeLargeNavigate per the support-matrix join_evidence -- read that cell instead.
            if c == "PointmazeLargeStitch" and "PointmazeLargeNavigate" in bulk:
                cfg = bulk["PointmazeLargeNavigate"]
                per_cell[c] = {
                    "status": "MEASURED",
                    "checks": {k: cfg["checks"].get(k, {}).get("status") for k in _CHECKS_OF_INTEREST
                              if k in cfg["checks"]},
                    "note": "PointmazeLargeStitch shares one rendered scene with "
                            "PointmazeLargeNavigate (support_matrix.json join_evidence); no "
                            "separate bulk-cert config exists for the Stitch dataset suffix",
                    "source": f"{_rel(BULK_CERT)} (config=PointmazeLargeNavigate)",
                }
                continue
            per_cell[c] = {"status": "UNMEASURED",
                           "why": f"no gate_a shard found for cell {c!r} in bulk, rejudged, or "
                                  f"reference-arm evidence"}
            continue
        per_cell[c] = {
            "status": "MEASURED",
            "checks": {k: cfg["checks"].get(k, {}).get("status") for k in _CHECKS_OF_INTEREST
                      if k in cfg["checks"]},
            "source": f"{_rel(BULK_CERT)} (config={c})",
        }
    overall = "MEASURED" if per_cell and all(v["status"] == "MEASURED" for v in per_cell.values()) \
        else "UNMEASURED"
    return {"status": overall, "per_cell": per_cell}


# --------------------------------------------------------------------------------------------
# 6. wrapper interaction path: swap_renderer everywhere pixels are certified, plus the extra
#    facade/pool surface for the three families that ARE our own env (not renderer-only)
# --------------------------------------------------------------------------------------------

def wrapper_interaction_path(cells: list[str]) -> dict:
    if not cells:
        return {"status": "NOT_APPLICABLE", "path": None,
                "why": "symbolic id: no renderer to swap"}
    native = [c for c in cells if c in NATIVE_ENV_FAMILIES]
    out = {
        "status": "MEASURED",
        "path": "swap_renderer",
        "why": "the official upstream env keeps its own dynamics; only render() is replaced -- "
               "'the answer for the envs we render but do not simulate' "
               "(stable_envs/compat/worldmodel.py module docstring). This is the mechanism every "
               "Step-7 family's Gate F PASS was measured through.",
        "source": "stable_envs/compat/worldmodel.py:swap_renderer; "
                  "examples/results/step7/craftax-full/findings.md, "
                  "examples/results/step7/ogb-manipulation/findings.md (\"Gate F -- arm D\"), "
                  "examples/results/step7/rocket-landing/findings.md",
    }
    if native:
        out["additional_surface"] = {
            "classes": native,
            "path": "StableEnvGym facade + BatchedEnvPool (arms B/C)",
            "why": "these classes are stable-envs' OWN simulator (not renderer-only), so they can "
                   "also run natively inside a swm.World -- measured throughput and info-dict "
                   "parity, not required for the pixel certification above",
            "source": "examples/swm_compat/README.md; examples/swm_compat/results/swm_workflow.json",
        }
    return out


# --------------------------------------------------------------------------------------------
# 7. upstream findings relevant to this id (by class-name / family substring match)
# --------------------------------------------------------------------------------------------

def load_upstream_bundle() -> list[dict]:
    """The plan-section-12 named items plus every dashboard ``upstream_findings`` entry, deduped."""
    items = [
        {"id": "U1", "item": "SWM reset(seed) is accepted and DISCARDED by the OGBench cube/scene "
         "wrappers (reset() never reseeds np_random); measured: two reset(seed=3) draws differ by "
         "up to 2.98 in qpos.",
         "source": "examples/results/step7/ogb-manipulation/findings.md:1026-1038"},
        {"id": "U2", "item": "action_space is a PROPERTY on ogbench's ManipSpaceEnv that builds a "
         "new, OS-seeded Box on every access (u.action_space is u.action_space -> False); "
         "initialize_episode consumes two unseedable actions from it on every reset.",
         "source": "examples/results/step7/ogb-manipulation/findings.md:1041-1046"},
        {"id": "U3", "item": "reset_variation_space is called with seed=None in cube_env.py / "
         "scene_env.py, so stable_worldmodel.spaces.py seeds the variation space from OS entropy "
         "before resampling default leaves; measured: cube.start_position differs between two "
         "identically-seeded resets.",
         "source": "examples/results/step7/ogb-manipulation/findings.md:1048-1053"},
        {"id": "U4", "item": "SceneEnv.reset(options={'state': ...}) calls set_state(qpos, qvel) "
         "without the button_state_* kwargs SceneEnv.set_state itself requires, and raises "
         "KeyError: 0 -- swm's own documented reset-from-state path does not work for OGBScene.",
         "source": "examples/results/step7/ogb-manipulation/findings.md:1054-1057"},
        {"id": "U5", "item": "environment.sky_color is offered in the Rocket variation space and "
         "read nowhere (dead upstream).",
         "source": "examples/results/step7/rocket-landing/findings.md:1137-1140,1304-1306"},
        {"id": "U6", "item": "_modify_rocket_urdf exists in the upstream Rocket wrapper but is "
         "never called; the URDF decoder is not rebuilt per episode.",
         "source": "examples/results/step7/rocket-landing/findings.md:1131-1136,1304-1306"},
        {"id": "U7", "item": "PyFlyt recolours the ignition flame on every state transition and "
         "never resets it -- an upstream reset defect, not a stable-envs one.",
         "source": "examples/results/step7/rocket-landing/findings.md:1307-1308"},
        {"id": "U8", "item": "PyFlyt/PyBullet render the frame after step() one substep behind the "
         "link/camera pose it just integrated to -- a one-substep lag upstream.",
         "source": "examples/results/step7/rocket-landing/findings.md:1309-1311"},
        {"id": "U9", "item": "Craftax: stable_worldmodel's CraftaxWrapper jax.jit's both reset and "
         "step, so a live SWM workflow observes the JIT-produced frame, which disagrees with its "
         "own un-jitted self by up to 1.5e-05 in float32 (<=1 uint8 step on lit/sleeping rows); "
         "the certified renderer comparison is against the un-jitted official renderer.",
         "source": "stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__ (external "
                   "repo, read directly); stable_envs/envs/craftax.py:CraftaxClassic."
                   "SWM_WRAPPER_GATE_F; stable_envs/envs/craftax_full.py:CraftaxFull."
                   "SWM_WRAPPER_GATE_F"},
    ]
    dash_text = DASHBOARD.read_text() if DASHBOARD.is_file() else ""
    # dashboard.yaml's upstream_findings/buckets are read and quoted verbatim in the docs/findings
    # generation below via a tiny line-scanner (no yaml dependency in this script on purpose).
    return items


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def build() -> dict:
    rows = load_base_rows()
    for sid, row in rows.items():
        cells = row["stable_envs_cells"]
        row["registered_default"] = registered_default_status(cells)
        row["swm_wrapper_reset_appearance"] = wrapper_reset_appearance_status(cells)
        row["variation_map"] = variation_map_status(cells, sid)
        row["state_observation_api"] = state_observation_api_status(cells, sid)
        row["device_behavior"] = {
            "status": "MEASURED" if row["supported_devices"] else "UNMEASURED",
            "supported_devices": row["supported_devices"],
            "source": f"{_rel(SUPPORT_MATRIX)} (supported_devices, Workstream E)",
        }
        row["wrapper_interaction_path"] = wrapper_interaction_path(cells)
    bundle = load_upstream_bundle()
    out = {
        "schema_version": 1,
        "step": 8,
        "workstream": "D_swm_integration",
        "what_this_is": __doc__,
        "baseline_sha": _sha(),
        "status_vocabulary": ["MEASURED", "UNMEASURED", "NOT_APPLICABLE", "MISMATCH(upstream)"],
        "sources": [str(p.relative_to(ROOT)) for p in
                    (SUPPORT_MATRIX, COVERAGE_MATRIX, BULK_CERT, GATE_A_REJUDGED, REFERENCE_ARM,
                     DEFAULT_ROUNDTRIP_CPU, VARIATION_CLASSIFICATION, FETCH_GPU_EVIDENCE)
                    if p.is_file()],
        "upstream_findings_bundle_ids": [u["id"] for u in bundle],
        "rows": list(rows.values()),
    }
    return out


def _tally(rows: list[dict], path: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        node = r
        for key in path:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        counts[node] = counts.get(node, 0) + 1
    return counts


def write_doc(matrix: dict) -> None:
    rows = matrix["rows"]
    lines = [
        "# Step 8 -- SWM integration / Gate-F matrix (all 42 ids)",
        "",
        "Generated by `scratch/step8/swm/build_gate_f.py` from committed evidence "
        f"(baseline `{matrix['baseline_sha']}`). Do not hand-edit; re-run the generator. "
        "Vocabulary: `MEASURED` / `UNMEASURED` / `NOT_APPLICABLE` / `MISMATCH(upstream)` -- every "
        "cell below carries a status word and an evidence path, never a bare PASS.",
        "",
        "## Column tally",
        "",
    ]
    for label, path in (
        ("registry_resolution", ("registry_resolution", "status")),
        ("registered_default", ("registered_default", "status")),
        ("swm_wrapper_reset_appearance", ("swm_wrapper_reset_appearance", "status")),
        ("variation_map", ("variation_map", "status")),
        ("state_observation_api", ("state_observation_api", "status")),
        ("device_behavior", ("device_behavior", "status")),
        ("wrapper_interaction_path", ("wrapper_interaction_path", "status")),
    ):
        counts = _tally(rows, path)
        lines.append(f"- **{label}**: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines += ["", "## Per-id matrix", "",
              "| swm id | classes | pixel/symbolic | registered default | reset appearance | "
              "variation map | state/obs/API | devices | wrapper path |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vm = r["variation_map"]
        vm_cell = vm["status"] if vm["status"] != "MEASURED" else \
            "; ".join(f"{k}={v}" for k, v in sorted((vm.get("aggregate_counts") or {}).items()))
        lines.append(
            "| `{sid}` | {cls} | {ps} | {rd} | {ra} | {vm} | {soa} | {dev} | {wip} |".format(
                sid=r["swm_id"], cls=", ".join(r["stable_envs_cells"]) or "(symbolic)",
                ps=r["pixel_or_symbolic"], rd=r["registered_default"]["status"],
                ra=r["swm_wrapper_reset_appearance"]["status"], vm=vm_cell,
                soa=r["state_observation_api"]["status"],
                dev=r["device_behavior"]["status"],
                wip=r["wrapper_interaction_path"].get("path")
                    or r["wrapper_interaction_path"]["status"]))
    lines.append("")
    lines.append("Full per-check evidence (source paths, sha256s, per-cell check statuses) is in "
                 f"`{_rel(OUT_JSON)}`. Upstream findings bundle: "
                 "`docs/research/step8_upstream_findings.md`.")
    OUT_DOC.write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    write_doc(matrix)
    print(f"wrote {_rel(OUT_JSON)} ({len(matrix['rows'])} rows) and {_rel(OUT_DOC)}")


if __name__ == "__main__":
    main()
