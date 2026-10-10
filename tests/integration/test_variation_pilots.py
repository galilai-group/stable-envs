"""The two variation hooks against pinned stable_worldmodel, one pilot leaf each."""

import pytest

pytest.importorskip("stable_worldmodel")

from tests.helpers.reference.gates.variation_pilots import PHYSICS_ATOL, physics_gate, reset_leaf_gate  # noqa: E402


@pytest.mark.parametrize("domain", ["two_room", "simple_point_maze"])
def test_agent_position_reset_distribution_matches_swm(domain):
    if domain == "simple_point_maze":
        matplotlib = pytest.importorskip("matplotlib")
        if matplotlib.__version__ != "3.10.9" or matplotlib.get_backend().lower() != "agg":
            pytest.skip("SimplePointMaze frames are certified with matplotlib 3.10.9 on the Agg backend")
    evidence = reset_leaf_gate(domain)
    assert evidence["explicit"]["state_exact"] and evidence["sampled"]["state_exact"]


def test_gravity_through_the_physics_hook_matches_swm():
    pytest.importorskip("dm_control")
    evidence = physics_gate()
    for case in ("explicit_gravity_z", "sampled_gravity_z", "restored_gravity_vector"):
        assert evidence[case]["max_state_error"] <= PHYSICS_ATOL
    assert evidence["sensitivity"]["shift_over_tolerance"] > 1e6
