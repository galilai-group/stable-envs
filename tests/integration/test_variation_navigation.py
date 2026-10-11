"""Navigation variation leaves against pinned stable_worldmodel."""

import pytest

pytest.importorskip("stable_worldmodel")

from tests.helpers.reference.gates.variation_navigation import (  # noqa: E402
    GATES,
    appearance_gate,
    min_steps_probe,
    reset_leaf_gate,
    zones_bias_field_gate,
)


@pytest.mark.parametrize(("domain", "leaf"), GATES)
def test_reset_leaf_matches_swm(domain, leaf):
    if domain == "simple_point_maze":
        matplotlib = pytest.importorskip("matplotlib")
        if matplotlib.__version__ != "3.10.9" or matplotlib.get_backend().lower() != "agg":
            pytest.skip("SimplePointMaze frames are certified with matplotlib 3.10.9 on the Agg backend")
    evidence = reset_leaf_gate(domain, leaf)
    assert evidence["explicit"]["state_exact"] and evidence["sampled"]["state_exact"]


def test_piecewise_target_and_bias_field_match_swm():
    pytest.importorskip("matplotlib")
    assert all(appearance_gate()["px_moved_by_leaves_per_lane"])
    assert zones_bias_field_gate()["first_frame"]["interior_bad_px"] == 0


def test_min_steps_is_dead_upstream():
    assert all(case["px_moved"] == 0 for case in min_steps_probe().values())
