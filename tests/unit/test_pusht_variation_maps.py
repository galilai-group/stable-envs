"""PushT and PushTDiscrete: every swm leaf is classified exactly once.

The static tests run against the leaf inventory recorded below. The live test compares that
inventory with the variation space of the installed stable-worldmodel.
"""

from __future__ import annotations

import pytest

from tests.helpers import reference as WM

SWM_LEAVES = (
    "agent.angle",
    "agent.color",
    "agent.scale",
    "agent.shape",
    "agent.start_position",
    "agent.velocity",
    "background.color",
    "block.angle",
    "block.color",
    "block.scale",
    "block.shape",
    "block.start_position",
    "goal.angle",
    "goal.color",
    "goal.position",
    "goal.scale",
    "rendering.render_goal",
)

CLASSES = {"PushT": "swm/PushT-v1", "PushTDiscrete": "swm/PushT-Discrete-v1"}
BUCKETS = {"mapped", "partial", "swm_only", "dead_upstream", "unsupported", "not_applicable"}


def test_the_recorded_leaf_count():
    assert len(SWM_LEAVES) == len(set(SWM_LEAVES)) == 17


@pytest.mark.parametrize(("cls", "swm_id"), sorted(CLASSES.items()))
def test_every_swm_leaf_is_classified_exactly_once(cls, swm_id):
    m = WM.variation_map(cls)
    assert m is not None and m.swm_id == swm_id
    leaves = set(SWM_LEAVES)
    mapped = m.mapped_swm_paths()
    declared = set(m.unmapped())
    assert not (mapped & declared), f"{cls}: mapped and declared unmapped: {sorted(mapped & declared)}"
    assert leaves - mapped - declared == set(), f"{cls}: undocumented {sorted(leaves - mapped - declared)}"
    assert mapped - leaves == set(), f"{cls}: maps leaves swm does not offer {sorted(mapped - leaves)}"
    assert declared - leaves == set(), f"{cls}: declares leaves swm does not offer {sorted(declared - leaves)}"
    assert mapped == leaves, f"{cls}: {sorted(leaves - mapped)} are not mapped"
    classification = m.classification()
    assert set(classification) == leaves
    assert set(classification.values()) == {"mapped"}


@pytest.mark.swm
@pytest.mark.parametrize("swm_id", sorted(CLASSES.values()))
def test_the_recorded_leaves_are_still_what_swm_offers(swm_id):
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_worldmodel")
    env = gymnasium.make(swm_id)
    try:
        live = sorted(WM.swm_variation_space(env).names())
    finally:
        env.close()
    assert live == sorted(SWM_LEAVES)
