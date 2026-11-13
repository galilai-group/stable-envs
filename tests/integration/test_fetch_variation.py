"""Fetch reset variation leaves against pinned stable_worldmodel."""

import pytest

pytestmark = pytest.mark.swm
pytest.importorskip("gymnasium_robotics")

from tests.helpers.reference.gates.fetch_variation import fetch_variation_gate  # noqa: E402


@pytest.mark.parametrize("scene", ["Reach", "Push", "Slide", "PickAndPlace"])
def test_reset_leaves_match_swm(scene):
    evidence = fetch_variation_gate(scene)
    for case in ("explicit", "sampled"):
        assert evidence[case]["state_exact"] and evidence[case]["appearance_exact"]
