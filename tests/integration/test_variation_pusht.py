"""PushT variation leaves against pinned stable_worldmodel."""

from importlib.metadata import PackageNotFoundError, version

import pytest

pytestmark = pytest.mark.swm
pytest.importorskip("pygame")
try:
    if version("pymunk") != "7.2.0":
        pytest.skip("PushT replay requires pymunk==7.2.0", allow_module_level=True)
except PackageNotFoundError:
    pytest.skip("PushT requires pymunk", allow_module_level=True)

from tests.helpers.reference.gates.variation_pusht import (  # noqa: E402
    CLASSES,
    EXPLICIT,
    leaf_gate,
    set_variation_gate,
)


@pytest.mark.parametrize("leaf", sorted(EXPLICIT))
@pytest.mark.parametrize("name", sorted(CLASSES))
def test_leaf_matches_swm_per_lane(name, leaf):
    evidence = leaf_gate(name, leaf)
    assert evidence["explicit"]["reset_frame_mse"] == 0.0 and evidence["sampled"]["reset_frame_mse"] == 0.0


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_set_variation_applies_appearance_now_and_the_rest_at_reset(name):
    evidence = set_variation_gate(name)
    assert evidence["immediate_frame_mse"] == evidence["next_reset_frame_mse"] == 0.0
