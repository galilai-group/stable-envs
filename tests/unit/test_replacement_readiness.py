"""Replacement-readiness overlay on VariationMap buckets (buckets themselves stay unchanged)."""

import re
from pathlib import Path

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import (
    STATUSES,
    inventory,
    status_counts,
    status_for,
)

pytestmark = pytest.mark.unit

REPORT = Path(__file__).resolve().parents[2] / "docs" / "research" / "step9_replacement_readiness.md"


def test_pusht_agent_color_is_native():
    assert VARIATION_MAPS["PushT"].replacement_class()["agent.color"] == "NATIVE"


def test_nav_speed_and_piecewise_bias_are_native():
    assert VARIATION_MAPS["TwoRoom"].replacement_class()["agent.speed"] == "NATIVE"
    assert VARIATION_MAPS["Piecewise"].replacement_class()["agent.speed"] == "NATIVE"
    assert VARIATION_MAPS["SimplePointMaze"].replacement_class()["agent.speed"] == "NATIVE"
    for i in range(4):
        assert VARIATION_MAPS["Piecewise"].replacement_class()[f"zones.bias_{i}"] == "NATIVE"


def test_fetch_block_mass_is_unsupported_meaningful():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["block.mass"] == "not_applicable"
    assert m.replacement_class()["block.mass"] == "UNSUPPORTED_MEANINGFUL"


def test_fetch_camera_angle_delta_is_dead_upstream():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["camera.angle_delta"] == "dead_upstream"
    assert m.replacement_class()["camera.angle_delta"] == "DEAD_UPSTREAM"


def test_fetch_background_color_is_partial():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["background.color"] == "partial"
    assert m.replacement_class()["background.color"] == "PARTIAL"


def test_ready_forbids_meaningful_gaps():
    for row in inventory():
        if row["status"] != "READY":
            continue
        assert row["counts"]["UNSUPPORTED_MEANINGFUL"] == 0
        assert row["counts"]["PARTIAL"] == 0
        assert row["execution"] in {"NATIVE", "DELEGATED"}


def test_every_historical_cell_has_a_known_status():
    rows = inventory()
    assert len(rows) == 42
    assert all(r["status"] in STATUSES for r in rows)
    counts = status_counts()
    assert (
        counts.get("READY", 0) + counts.get("PARTIAL", 0) + counts.get("INCOMPLETE", 0) + counts.get("SYMBOLIC", 0)
        == 42
    )


def test_the_readiness_report_counts_match_the_inventory():
    text = REPORT.read_text()
    m = re.search(r"<!-- readiness-counts READY=(\d+) PARTIAL=(\d+) INCOMPLETE=(\d+) SYMBOLIC=(\d+) -->", text)
    assert m, "report is missing the machine-readable counts comment"
    got = {k: int(v) for k, v in zip(("READY", "PARTIAL", "INCOMPLETE", "SYMBOLIC"), m.groups())}
    counts = status_counts()
    for key in got:
        assert got[key] == counts.get(key, 0), f"{key}: report {got[key]} vs inventory {counts.get(key, 0)}"
    for row in inventory():
        assert f"`{row['swm_id']}`" in text
        # status in the table row must not contradict the derived status
        assert row["status"] == status_for(row["swm_id"])
