"""The two variation hooks against pinned stable_worldmodel, one pilot leaf each."""

import pytest

pytest.importorskip("stable_worldmodel")

from tests.helpers.reference.gates.variation_pilots import (  # noqa: E402
    PHYSICS_ATOL,
    classic_physics_gate,
    navigation_reset_rule_gate,
    physics_gate,
    reset_leaf_gate,
    reset_rule_gate,
)


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


def test_plain_reset_restores_defaults_as_swm_does():
    pytest.importorskip("dm_control")
    evidence = reset_rule_gate()
    for case in ("registered_defaults", "set_variation_defaults"):
        assert evidence[case]["max_state_error"] <= PHYSICS_ATOL


def test_plain_reset_restores_the_navigation_default_as_swm_does():
    assert navigation_reset_rule_gate()["steps"] > 0


@pytest.mark.parametrize("domain", ["cartpole", "pendulum", "acrobot", "mountain_car", "mountain_car_continuous"])
def test_classic_physics_matches_swm(domain):
    evidence = classic_physics_gate(domain)
    assert evidence["plain_reset_restores_defaults"]
    assert any(all(leaf["moved_vs_default"]) for leaf in evidence["leaves"].values())
