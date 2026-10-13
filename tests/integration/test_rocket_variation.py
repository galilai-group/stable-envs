"""PyFlyt Rocket start leaves and reset colours against pinned stable_worldmodel."""

import pytest

pytest.importorskip("PyFlyt")
pytestmark = pytest.mark.swm

from tests.helpers.reference.gates.variation_rocket import rocket_reset_leaf_gate  # noqa: E402


def test_rocket_start_leaves_and_reset_colours_match_swm():
    evidence = rocket_reset_leaf_gate()
    assert all(evidence[case]["exact"] for case in ("explicit", "sampled", "colours"))
