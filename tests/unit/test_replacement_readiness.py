"""Replacement-readiness overlay on VariationMap buckets (buckets themselves stay unchanged)."""

import pytest

from tests.helpers.reference import VARIATION_MAPS

pytestmark = pytest.mark.unit


def test_pusht_agent_color_is_native():
    assert VARIATION_MAPS["PushT"].replacement_class()["agent.color"] == "NATIVE"


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
