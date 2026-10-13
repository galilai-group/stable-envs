"""Cube/Scene Variation leaves against pinned stable_worldmodel, per lane."""

import pytest

pytestmark = pytest.mark.swm
pytest.importorskip("ogbench")

from tests.helpers.reference.gates.ogb_variation import reset_options_gate, set_variation_gate  # noqa: E402


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_reset_option_leaves_match_swm_per_lane(family):
    assert not reset_options_gate(family)["placement_leaves_read_in_task_mode"]


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_set_variation_applies_now_and_resets_as_swm(family):
    assert set_variation_gate(family)["next_reset_matches_swm"]
