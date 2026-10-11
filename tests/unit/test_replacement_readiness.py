"""Replacement-readiness overlay on VariationMap buckets (buckets themselves stay unchanged)."""

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import (
    STATUSES,
    inventory,
    status_counts,
)

pytestmark = pytest.mark.unit


def test_pusht_agent_color_is_native():
    assert VARIATION_MAPS["PushT"].replacement_class()["agent.color"] == "NATIVE"


def test_nav_speed_and_piecewise_bias_are_native():
    assert VARIATION_MAPS["TwoRoom"].replacement_class()["agent.speed"] == "NATIVE"
    assert VARIATION_MAPS["Piecewise"].replacement_class()["agent.speed"] == "NATIVE"
    assert VARIATION_MAPS["SimplePointMaze"].replacement_class()["agent.speed"] == "NATIVE"
    for i in range(4):
        assert VARIATION_MAPS["Piecewise"].replacement_class()[f"zones.bias_{i}"] == "NATIVE"


def test_fetch_block_mass_is_native():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["block.mass"] == "mapped"
    assert m.replacement_class()["block.mass"] == "NATIVE"


def test_fetch_camera_angle_delta_is_dead_upstream():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["camera.angle_delta"] == "dead_upstream"
    assert m.replacement_class()["camera.angle_delta"] == "DEAD_UPSTREAM"


def test_fetch_background_color_is_native():
    m = VARIATION_MAPS["FetchPush"]
    assert m.classification()["background.color"] == "mapped"
    assert m.replacement_class()["background.color"] == "NATIVE"


def test_ready_forbids_meaningful_gaps():
    for row in inventory():
        if row["status"] != "READY":
            continue
        assert row["counts"]["UNSUPPORTED_MEANINGFUL"] == 0
        assert row["counts"]["PARTIAL"] == 0
        assert row["execution"] in {"NATIVE", "DELEGATED_CPU"}


def test_every_historical_cell_has_a_known_status():
    rows = inventory()
    assert len(rows) == 42
    assert all(r["status"] in STATUSES for r in rows)
    counts = status_counts()
    assert (
        counts.get("READY", 0) + counts.get("PARTIAL", 0) + counts.get("INCOMPLETE", 0) + counts.get("SYMBOLIC", 0)
        == 42
    )
