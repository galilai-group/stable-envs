"""Every DM Control physics Variation leaf against the pinned stable_worldmodel wrappers."""

import pytest

pytest.importorskip("dm_control")
pytestmark = pytest.mark.swm

from tests.helpers.reference.gates.dmc_physics import (  # noqa: E402
    CELLS,
    INERT_UPSTREAM,
    PHYSICS_TOL,
    reset_rule_gate,
    run_gate,
)


@pytest.mark.parametrize("cell", CELLS)
def test_physics_leaves_match_swm(cell):
    evidence = run_gate([cell])[cell]
    for leaf, case in evidence["leaves"].items():
        assert case["max_state_error"] <= PHYSICS_TOL, (leaf, case)
        if leaf.startswith("gravity.") or leaf.endswith(".friction"):
            assert case["max_state_error"] == 0.0, (leaf, case)
        assert case["sensitive"] or (cell, leaf) in INERT_UPSTREAM, (leaf, case)
    sampled = evidence["sampled_and_restored"]
    assert max(sampled["max_state_error"], sampled["restored_max_state_error"]) <= PHYSICS_TOL, sampled


def test_plain_reset_restores_defaults_as_swm_does():
    evidence = reset_rule_gate()
    for case in ("registered_defaults", "set_variation_defaults"):
        assert evidence[case]["max_state_error"] == 0.0, evidence
