"""Classic-control physics Variation leaves against pinned stable_worldmodel."""

import pytest

pytestmark = pytest.mark.swm

from tests.helpers.reference.gates.variation_classic import classic_physics_gate  # noqa: E402


@pytest.mark.parametrize("domain", ["cartpole", "pendulum", "acrobot", "mountain_car", "mountain_car_continuous"])
def test_classic_physics_matches_swm(domain):
    evidence = classic_physics_gate(domain)
    assert evidence["plain_reset_restores_defaults"]
    assert any(all(leaf["moved_vs_default"]) for leaf in evidence["leaves"].values())
