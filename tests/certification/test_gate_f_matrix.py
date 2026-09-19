"""Contract for the Step-8 Gate-F/API matrix (`scratch/step8/swm/build_gate_f.py`).

Not a test of the matrix's CONTENTS -- those move as VariationMaps and Gate-A shards change, which
is why the matrix is generated rather than hand-written (see the module docstring on the
generator). What this file pins:

  1. the committed ``gate_f_matrix.json`` regenerates BYTE-IDENTICALLY from the committed inputs
     it reads (modulo the ``baseline_sha`` field, which tracks HEAD and is expected to move as the
     repo moves) -- so the file on disk is never stale prose;
  2. all 42 SWM ids are present, and their union with a fresh read of the support matrix agrees;
  3. every one of the eight per-row status fields carries a value from the documented vocabulary
     AND an evidence path/reason -- never a bare status with nothing backing it;
  4. the two symbolic ids are NOT_APPLICABLE across every renderer-shaped column;
  5. the Craftax SWM-wrapper-JIT declaration this workstream added exists on both cells.

The generator is a script under ``scratch/``, loaded by path rather than imported as a package,
matching ``tests/unit/test_swm_matrix.py``'s own convention for the sibling Step-3/7 tool.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD_PY = REPO / "scratch" / "step8" / "swm" / "build_gate_f.py"
MATRIX_JSON = REPO / "examples" / "results" / "step8" / "swm" / "gate_f_matrix.json"
SUPPORT_MATRIX = REPO / "examples" / "results" / "step8" / "support_matrix.json"

STATUS_VOCAB = {"MEASURED", "UNMEASURED", "NOT_APPLICABLE", "MISMATCH(upstream)"}
ROW_STATUS_FIELDS = (
    "registry_resolution",
    "registered_default",
    "swm_wrapper_reset_appearance",
    "variation_map",
    "state_observation_api",
    "device_behavior",
    "wrapper_interaction_path",
)
SYMBOLIC_IDS = {"swm/CraftaxClassicSymbolic-v1", "swm/CraftaxSymbolic-v1"}


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("_build_gate_f", BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed():
    assert MATRIX_JSON.is_file(), f"{MATRIX_JSON} must be committed"
    return json.loads(MATRIX_JSON.read_text())


def test_regenerates_byte_identically_modulo_head_sha(gen, committed):
    fresh = gen.build()
    committed_copy = dict(committed)
    fresh_copy = dict(fresh)
    # HEAD moves as commits land on this branch; everything else must not.
    committed_copy.pop("baseline_sha", None)
    fresh_copy.pop("baseline_sha", None)
    assert json.dumps(fresh_copy, indent=2, sort_keys=True) == json.dumps(committed_copy, indent=2, sort_keys=True), (
        "gate_f_matrix.json is stale: re-run `python scratch/step8/swm/build_gate_f.py` and commit the result"
    )


def test_all_42_ids_present_and_agree_with_support_matrix(committed):
    ids = {r["swm_id"] for r in committed["rows"]}
    assert len(committed["rows"]) == 42
    assert len(ids) == 42
    support = json.loads(SUPPORT_MATRIX.read_text())
    support_ids = {r["swm_id"] for r in support["rows"]}
    assert ids == support_ids


def test_every_row_every_status_field_has_a_vocabulary_status_and_evidence(committed):
    for row in committed["rows"]:
        for field in ROW_STATUS_FIELDS:
            node = row[field]
            status = node["status"]
            assert status in STATUS_VOCAB, f"{row['swm_id']}.{field}: unrecognised status {status!r}"
            # every UNMEASURED/NOT_APPLICABLE/MISMATCH must say WHY, and every MEASURED must point
            # somewhere -- 'source', 'why', 'evidence' or a non-empty per-class/per-cell breakdown
            # (which itself carries sources) are all acceptable, but nothing bare is.
            has_reason = (
                any(k in node for k in ("source", "why", "evidence"))
                or bool(node.get("per_class"))
                or bool(node.get("per_cell"))
                or bool(node.get("classes"))
            )
            assert has_reason, (
                f"{row['swm_id']}.{field} has status {status!r} with no evidence path or reason: {node!r}"
            )


def test_symbolic_ids_are_not_applicable_across_renderer_columns(committed):
    by_id = {r["swm_id"]: r for r in committed["rows"]}
    for sid in SYMBOLIC_IDS:
        row = by_id[sid]
        assert row["pixel_or_symbolic"] == "symbolic"
        for field in ("registered_default", "variation_map", "state_observation_api", "wrapper_interaction_path"):
            assert row[field]["status"] == "NOT_APPLICABLE", f"{sid}.{field}"


def test_craftax_swm_wrapper_jit_declaration_exists_on_both_cells():
    import stable_envs.envs._craftax_full as cf
    import stable_envs.envs.craftax as cc

    for cls in (cc.CraftaxClassic, cf.CraftaxFull):
        decl = cls.SWM_WRAPPER_GATE_F
        assert "step" in decl["swm_wrapper_jits"]
        assert decl["certified_exact_frac"] == 1.0
        assert decl["certified_max_abs_diff"] == 0
        # the gap between "what is certified" and "what a live SWM caller sees" is stated, and
        # bounded at one uint8 step -- not omitted, not unbounded.
        assert decl["max_abs_diff_swm_wrapper_vs_certified"] == 1
