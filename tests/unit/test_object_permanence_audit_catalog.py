"""Measured source coverage must account for every nominal and extra member."""

import json
import re
from pathlib import Path

from scripts.export.object_permanence import (
    ALPHA_TASKS,
    AUDITED_TASKS,
    OPTICAL_TASKS,
    ORTHOGRAPHIC_TASKS,
    RAY_CONTRACT_TASKS,
    SOURCE_COMMIT,
)
from tests.helpers.reference.object_permanence_recordings import DEPTH_MSE_MAX, MASK_IOU_MIN, RECORDINGS

CATALOG = Path(__file__).resolve().parents[2] / "docs/adding/object-permanence-audit.json"


def test_complete_nominal_audit_accounts_for_all_source_members_without_pending_cases():
    audit = json.loads(CATALOG.read_text())
    assert audit["schema"] == "stable-envs/object-permanence-source-audit-v1"
    assert audit["source_commit"] == SOURCE_COMMIT
    assert audit["profile"] == "first-nominal-member"
    entries = audit["entries"]
    assert len(entries) == 150
    assert {r["task"] for r in entries} == {f"G{i:02}" for i in range(1, 151)}
    assert all(r["member"] == 0 and r["status"] in ("certified", "deferred") for r in entries)
    extras = audit["unregistered_members"]
    assert len(extras) == 43 and len({(r["task"], r["member"]) for r in extras}) == 43
    additional = audit["additional_certified_members"]
    assert {(r["task"], r["member"]) for r in additional} == {("G01", 1), ("G02", 1), ("G02", 2), ("G03", 1)}
    assert all(r["status"] == "certified" for r in additional)
    assert all(r["member"] > 0 and r["status"] == "unregistered-member-entry" for r in extras)
    assert len(entries) + len(extras) + len(additional) == audit["scope"]["member_scripts"] == 197
    for row in entries + extras + additional:
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_script_sha256"])
        assert row["script"].startswith("object_permanence/generator/tasks/")


def test_certified_catalog_matches_registered_recordings_and_unchanged_budgets():
    audit = json.loads(CATALOG.read_text())
    assert audit["thresholds"] == {"depth_mse_max": DEPTH_MSE_MAX, "mask_iou_min": MASK_IOU_MIN}
    certified = {r["task"]: r for r in audit["entries"] if r["status"] == "certified"}
    assert set(certified) == set(AUDITED_TASKS) | OPTICAL_TASKS
    for task, row in certified.items():
        path = RECORDINGS / "clips" / f"{task.lower()}-nominal/manifest.json"
        recording = json.loads(path.read_text())
        assert recording["source"]["script"] == row["script"]
        assert recording["source"]["script_sha256"] == row["source_script_sha256"]
        assert recording["source"]["task_manifest_sha256"] == row["source_task_manifest_sha256"]
        assert row["frames"] == 120
        assert (row["contract"] is not None) == (
            task in RAY_CONTRACT_TASKS | ORTHOGRAPHIC_TASKS | ALPHA_TASKS | OPTICAL_TASKS
        )
        if task in OPTICAL_TASKS:
            assert recording["schema"] == "stable-envs/object-permanence-rigid-v5"
            assert row["contract"] == recording["visibility_contract"] == "stable-envs/sampled-optical-rays-v1"
            assert row["native_bsdf_pixel_equivalence_claimed"] is False
        if task in ALPHA_TASKS:
            assert recording["schema"] == "stable-envs/object-permanence-rigid-v4"
            assert row["contract"] == recording["visibility_contract"] == "stable-envs/transparent-pixel-rays-v1"
        if task in ORTHOGRAPHIC_TASKS:
            assert row["contract"] == "stable-envs/opaque-pixel-rays-v2"
            assert recording["schema"] == "stable-envs/object-permanence-rigid-v2"
            assert recording["camera"]["projection_type"] == "orthographic"
        if "depth_mse_max" in row:
            assert row["depth_mse_max"] <= DEPTH_MSE_MAX
            assert row["mask_iou_min"] >= MASK_IOU_MIN
            assert row["cache_bit_identical_resolutions"] == [90, 180]
            version = "v2" if task in ORTHOGRAPHIC_TASKS else "v1"
            oracle = RECORDINGS.parents[1] / (
                f"semantics/object_permanence/{task.lower()}/sampled_optical_rays_v1/nominal/manifest.json"
                if task in OPTICAL_TASKS
                else f"semantics/object_permanence/{task.lower()}/transparent_pixel_rays_v1/nominal/manifest.json"
                if task in ALPHA_TASKS
                else f"semantics/object_permanence/{task.lower()}/opaque_pixel_rays_{version}/nominal/manifest.json"
            )
            assert json.loads(oracle.read_text())["targets"] == row["reference_targets"]
            roles = {o["name"]: o["role"] for o in recording["objects"]}
            assert {name: roles[name] for name in row["reference_targets"]} == row["reference_target_roles"]


def test_deferred_catalog_never_exposes_an_accepted_recording_or_oracle():
    audit = json.loads(CATALOG.read_text())
    fixtures = Path(__file__).resolve().parents[1] / "fixtures/semantics/object_permanence"
    for row in audit["entries"]:
        if row["status"] != "deferred":
            continue
        key = row["task"].lower()
        assert row["category"] and row["reason"]
        assert not (RECORDINGS / "clips" / f"{key}-nominal").exists()
        assert not (fixtures / key / "opaque_pixel_rays_v1/nominal").exists()
        assert not (fixtures / key / "opaque_pixel_rays_v2/nominal").exists()
        assert row["preview"].startswith(".cache/object_permanence/evaluation/deferred-previews/")


def test_surface_context_audit_records_defaults_and_remaining_blockers_without_waivers():
    audit = json.loads(CATALOG.read_text())
    entries = {row["task"]: row for row in audit["entries"]}
    defaults = {
        "G08": "drawer_travel_scale",
        "G44": "ball_radius_scale",
        "G45": "ball_radius_scale",
        "G46": "cover_footprint_scale",
        "G48": "object_spacing_scale",
    }
    for task, parameter in defaults.items():
        row = entries[task]
        context = row["execution_context"]
        assert context["profile"] == "surface"
        assert context["globals"] == {"DIVERSITY": {}} and context["source_bindings"] == {}
        assert row["diversity_defaults"] == {parameter: 1.0}
        assert row["source_capability_audit"]["frames"] == row["preview_frames"] == 120
        recording = json.loads((RECORDINGS / f"clips/{task.lower()}-nominal/manifest.json").read_text())
        assert row["status"] == "certified" and row["camera"] == ("PERSP" if task == "G44" else "ORTHO")
        assert recording["source"]["execution_context"] == context
    assert entries["G46"]["source_capability_audit"]["unsupported_objects"] == {}
    assert entries["G46"]["source_capability_audit"]["converted_objects"] == {
        "box_A_label": "FONT",
        "box_B_label": "FONT",
    }
    assert "blockers" not in entries["G46"]


def test_additional_certified_members_keep_their_source_identity_and_versioned_oracle():
    audit = json.loads(CATALOG.read_text())
    fixtures = Path(__file__).parents[1] / "fixtures"
    for row in audit["additional_certified_members"]:
        clip = row["task"].lower() + f"-member{row['member']}"
        m = json.loads((RECORDINGS / "clips" / clip / "manifest.json").read_text())
        assert m["source"]["script_sha256"] == row["source_script_sha256"]
        assert m["source"]["task_manifest_sha256"] == row["source_task_manifest_sha256"]
        assert m["source"]["member"] == row["member"]
        assert m.get("visibility_contract", "stable-envs/opaque-pixel-rays-v1") == row["contract"]
        assert row["depth_mse_max"] <= DEPTH_MSE_MAX and row["mask_iou_min"] >= MASK_IOU_MIN
        version = row["contract"].split("/")[1].replace("-", "_")
        oracle = fixtures / f"semantics/object_permanence/{row['task'].lower()}/{version}/member{row['member']}"
        source = json.loads((oracle / "manifest.json").read_text())
        assert source["source_script_sha256"] == row["source_script_sha256"]
        assert source.get("frames", source["arrays"]["source_frames"]["shape"][0]) == row["frames"] == 120
