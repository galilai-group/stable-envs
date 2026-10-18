"""TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones: every swm leaf is classified exactly once.

The static tests run against the leaf inventory recorded below. The live test compares that
inventory with the variation space of the installed stable-worldmodel.
"""

from __future__ import annotations

import pytest

from tests.helpers import reference as WM

_AGENT = ("agent.color", "agent.position", "agent.radius", "agent.speed")
_TARGET = ("target.color", "target.position", "target.radius")

SWM_LEAVES = {
    "swm/TwoRoom-v1": tuple(
        sorted(
            _AGENT
            + _TARGET
            + (
                "background.color",
                "door.color",
                "door.number",
                "door.position",
                "door.size",
                "rendering.render_target",
                "task.min_steps",
                "wall.axis",
                "wall.border_color",
                "wall.color",
                "wall.thickness",
            )
        )
    ),
    # grid_n=2: four zones
    "swm/Piecewise-v0": tuple(
        sorted(
            _AGENT
            + _TARGET
            + (
                "background.color",
                "border.color",
                "rendering.render_bias_field",
                "rendering.render_target",
                "rendering.render_zones",
                "task.min_steps",
            )
            + tuple(f"zones.bias_{i}" for i in range(4))
            + tuple(f"zones.color_{i}" for i in range(4))
        )
    ),
}

CLASSES = {
    "swm/TwoRoom-v1": ("TwoRoom", "TwoRoomTarget"),
    "swm/Piecewise-v0": ("Piecewise", "PiecewiseZones"),
}
#: (leaves, leaves mapped by at least one of the two classes)
EXPECTED_COUNTS = {"swm/TwoRoom-v1": (18, 15), "swm/Piecewise-v0": (21, 19)}
EXPECTED_UNMAPPED = {
    "swm/TwoRoom-v1": {
        "task.min_steps": "dead_upstream",
        "rendering.render_target": "swm_only",
        "wall.border_color": "swm_only",
    },
    "swm/Piecewise-v0": {"task.min_steps": "dead_upstream", "rendering.render_zones": "swm_only"},
}
PAIRS = [(swm_id, cls) for swm_id, classes in CLASSES.items() for cls in classes]
BUCKETS = {"mapped", "partial", "swm_only", "dead_upstream", "unsupported", "not_applicable"}


@pytest.mark.parametrize(("swm_id", "cls"), PAIRS)
def test_every_swm_leaf_is_classified_exactly_once(swm_id, cls):
    m = WM.variation_map(cls)
    assert m is not None and m.swm_id == swm_id
    leaves = set(SWM_LEAVES[swm_id])
    mapped = m.mapped_swm_paths()
    declared = set(m.unmapped())
    assert not (mapped & declared), f"{cls}: mapped and declared unmapped: {sorted(mapped & declared)}"
    assert set(m.partial) <= mapped
    assert leaves - mapped - declared == set(), f"{cls}: undocumented {sorted(leaves - mapped - declared)}"
    assert mapped - leaves == set(), f"{cls}: maps leaves swm does not offer {sorted(mapped - leaves)}"
    assert declared - leaves == set(), f"{cls}: declares leaves swm does not offer {sorted(declared - leaves)}"
    classification = m.classification()
    assert set(classification) == leaves
    assert set(classification.values()) <= BUCKETS


@pytest.mark.parametrize("swm_id", sorted(CLASSES))
def test_the_two_classes_of_one_swm_id_cover_its_leaves(swm_id):
    maps = [WM.variation_map(c) for c in CLASSES[swm_id]]
    leaves = set(SWM_LEAVES[swm_id])
    total, n_mapped = EXPECTED_COUNTS[swm_id]
    mapped = set().union(*(m.mapped_swm_paths() for m in maps))
    assert len(leaves) == total
    assert len(mapped) == n_mapped
    assert leaves - mapped == set(EXPECTED_UNMAPPED[swm_id])
    for m in maps:
        classification = m.classification()
        for leaf, bucket in EXPECTED_UNMAPPED[swm_id].items():
            assert classification[leaf] == bucket, f"{leaf} in {swm_id}"


@pytest.mark.swm
@pytest.mark.parametrize("swm_id", sorted(SWM_LEAVES))
def test_the_recorded_leaves_are_still_what_swm_offers(swm_id):
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_worldmodel")
    env = gymnasium.make(swm_id)
    try:
        live = sorted(WM.swm_variation_space(env).names())
    finally:
        env.close()
    assert live == sorted(SWM_LEAVES[swm_id])
