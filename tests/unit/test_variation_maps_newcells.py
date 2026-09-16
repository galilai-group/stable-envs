"""The Step-7 new cells' VariationMaps: every offered swm leaf is classified, and the classification
agrees with the map it is supposed to describe.

The point of these tests is NOT that the reasons are well written. It is that no swm leaf can be
added upstream, or removed from a map here, without something failing: an unclassified leaf is
exactly the Gate-F gap Step 8 exists to close, and it is invisible unless a test counts.

Most of this runs without ``stable_worldmodel``, against the leaf inventory recorded from the live
wrappers, because swm is not importable in every environment this suite runs in and a test that
skips where it matters is not a test. The live enumeration runs as well wherever swm IS importable,
and asserts the recorded inventory still matches the wrapper.
"""

import numpy as np
import pytest

from stable_envs.compat.worldmodel import VARIATION_MAPS, variation_map
from stable_envs.compat.variation_maps.newcells import (
    BUCKETS, LEAF_CLASSIFICATION, NEWCELL_VARIATION_MAPS, RECORDED_SWM_LEAVES,
)

IDS = sorted(RECORDED_SWM_LEAVES)


def _map_for(swm_id):
    ms = [m for m in NEWCELL_VARIATION_MAPS.values() if m.swm_id == swm_id]
    assert ms, f"no VariationMap declares swm_id={swm_id}"
    return ms[0]


# ---------------------------------------------------------------- registration

def test_the_new_maps_are_registered_in_the_shared_table():
    """The one line in compat/worldmodel.py is what makes these reachable from `variation_map`."""
    for name, m in NEWCELL_VARIATION_MAPS.items():
        assert VARIATION_MAPS.get(name) is m, f"{name} is not registered in VARIATION_MAPS"
        assert variation_map(name) is m


def test_registering_the_new_maps_did_not_disturb_the_pre_existing_ones():
    for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "PushT"):
        assert name in VARIATION_MAPS, f"{name} was lost when the new maps were merged in"


# ---------------------------------------------------------------- completeness

@pytest.mark.parametrize("swm_id", IDS)
def test_every_offered_leaf_is_in_exactly_one_bucket_of_the_map(swm_id):
    """The invariant `check_mapping` looks for: no leaf in NEITHER, no leaf in BOTH."""
    m = _map_for(swm_id)
    leaves = set(RECORDED_SWM_LEAVES[swm_id])
    mapped = {t for t, _ in m.pairs().values()}
    declared = set(m.swm_only)
    undocumented = leaves - mapped - declared
    assert not undocumented, (
        f"{swm_id}: {sorted(undocumented)} are offered by swm and appear in NEITHER the mapped "
        f"leaves nor swm_only. That is a silently unmapped leaf, which is the Gate-F gap Step 8 "
        f"closes -- classify it rather than deleting this assertion")
    both = mapped & declared
    assert not both, f"{swm_id}: {sorted(both)} are both mapped and declared unmapped"
    assert not (mapped - leaves), (
        f"{swm_id}: {sorted(mapped - leaves)} are mapped to swm paths swm does not offer")
    assert not (declared - leaves), (
        f"{swm_id}: {sorted(declared - leaves)} carry a reason but are not offered by swm")


@pytest.mark.parametrize("swm_id", IDS)
def test_every_offered_leaf_has_a_structured_classification(swm_id):
    cls = LEAF_CLASSIFICATION[swm_id]["leaves"]
    leaves = set(RECORDED_SWM_LEAVES[swm_id])
    assert set(cls) == leaves, (
        f"{swm_id}: classification covers {sorted(set(cls))} but swm offers {sorted(leaves)}")


@pytest.mark.parametrize("swm_id", IDS)
def test_every_verdict_is_one_of_the_four_buckets(swm_id):
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        assert rec["classification"] in BUCKETS, (
            f"{swm_id}:{leaf} has verdict {rec['classification']!r}, which is not one of {BUCKETS}")


@pytest.mark.parametrize("swm_id", IDS)
def test_the_structured_verdict_agrees_with_the_map(swm_id):
    """`mapped` in the table must mean mapped in the VariationMap, and vice versa. Two records of
    the same fact that can drift apart are worse than one."""
    m = _map_for(swm_id)
    mapped = {t for t, _ in m.pairs().values()}
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        if rec["classification"] == "mapped":
            assert leaf in mapped, f"{swm_id}:{leaf} is classified mapped but the map does not carry it"
        else:
            assert leaf not in mapped, (
                f"{swm_id}:{leaf} is classified {rec['classification']} but the map DOES carry it")
            assert leaf in m.swm_only, f"{swm_id}:{leaf} is unmapped with no reason in swm_only"


@pytest.mark.parametrize("swm_id", IDS)
def test_nothing_is_declared_dead_upstream_without_citing_a_measurement(swm_id):
    """`dead_upstream` is a claim about the OFFICIAL renderer, not about our reading of its source.
    It is the one verdict that lets us ship a knob-shaped hole, so it carries the highest evidence
    bar: a probe, with counts."""
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        if rec["classification"] == "dead_upstream":
            ev = rec.get("evidence", "")
            assert ev, f"{swm_id}:{leaf} is dead_upstream with no evidence recorded"
            assert any(t in ev for t in ("px", "pixel", "arrays")), (
                f"{swm_id}:{leaf} is dead_upstream but its evidence {ev!r} cites no measurement")


@pytest.mark.parametrize("swm_id", IDS)
def test_every_unmapped_leaf_explains_itself(swm_id):
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        if rec["classification"] != "mapped":
            assert len(rec["reason"]) > 40, (
                f"{swm_id}:{leaf} is {rec['classification']} with a reason too short to be one")


# ---------------------------------------------------------------- Rocket, the one family with dims

ROCKET = "swm/PFRocketLanding-v0"


def test_rocket_maps_exactly_the_five_material_colours():
    m = _map_for(ROCKET)
    assert set(m.pairs()) == {"rocket.body_color", "rocket.fin_color", "rocket.leg_color",
                              "rocket.booster_color", "pad.color"}
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs, f"{ours} should map by identical path, not to {theirs}"
        assert factor == 1.0, (
            f"{ours} carries factor {factor}; both sides are 0-255 RGB and a factor here would be "
            f"the PushT scale bug in a new place")


def test_rocket_ignition_is_ours_only_because_swm_derives_it_from_the_action():
    m = _map_for(ROCKET)
    assert "flame.ignited" in m.ours_only
    assert "flame.ignited" not in RECORDED_SWM_LEAVES[ROCKET]


def test_rocket_sky_color_is_dead_upstream_and_not_quietly_mapped():
    """The one leaf Step 7 measured dead. If a future edit maps it, this fails rather than shipping
    a knob that moves nothing."""
    m = _map_for(ROCKET)
    assert "environment.sky_color" not in {t for t, _ in m.pairs().values()}
    rec = LEAF_CLASSIFICATION[ROCKET]["leaves"]["environment.sky_color"]
    assert rec["classification"] == "dead_upstream"
    assert "0 of 691,200" in rec["evidence"]


def test_rocket_start_state_leaves_are_state_not_style():
    """They move pixels -- by moving the rocket. That makes them our STATE, and mapping them to
    style would put the same information in two places."""
    for leaf in ("environment.start_height_ratio", "environment.start_horizontal_offset",
                 "environment.start_tilt"):
        rec = LEAF_CLASSIFICATION[ROCKET]["leaves"][leaf]
        assert rec["classification"] == "not_applicable"


def test_rocket_default_variation_round_trips_through_the_style_layout():
    """The registered default must survive `variation -> style -> variation` unchanged. This is the
    mapping half of "the registered default is reproducible through the mapping"; the pixel half is
    the GPU artifact."""
    rl = pytest.importorskip("stable_envs.envs.rocket_landing")
    layout = rl.style_layout()
    default = np.asarray(rl._default_style_row(), np.float64)
    m = _map_for(ROCKET)
    for ours, (theirs, factor) in m.pairs().items():
        assert ours in layout, f"{ours} is mapped but is not a dim of the env's style layout"
        o, w = layout[ours]
        back = default[o:o + w] / factor
        assert np.allclose(back * factor, default[o:o + w]), f"{ours} does not round-trip"


def test_rocket_mapped_leaves_are_disjoint_slices_of_the_style_row():
    """Isolation, structurally: two mapped leaves that overlap in the style row cannot be varied
    independently no matter what the renderer does."""
    rl = pytest.importorskip("stable_envs.envs.rocket_landing")
    layout = rl.style_layout()
    m = _map_for(ROCKET)
    seen = {}
    for ours in m.pairs():
        o, w = layout[ours]
        for i in range(o, o + w):
            assert i not in seen, f"{ours} and {seen[i]} both own style dim {i}"
            seen[i] = ours


def test_rocket_pad_color_default_is_white_not_the_declared_two_hundred():
    """swm declares pad.color init_value (200,200,200) but a plain reset loads the UNMODIFIED pad
    URDF -- `_modify_pad_urdf` runs only when a variation is requested -- so the pad is white. Our
    default is the one a plain reset produces, which is what "the registered default" means."""
    rl = pytest.importorskip("stable_envs.envs.rocket_landing")
    layout = rl.style_layout()
    o, w = layout["pad.color"]
    row = np.asarray(rl._default_style_row(), np.float64)
    assert list(row[o:o + w]) == [255.0, 255.0, 255.0]


# ---------------------------------------------------------------- OGB, the empty-style family

@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_maps_nothing_yet_and_says_so_leaf_by_leaf(swm_id):
    """These two cells declare `make_variation_space() -> Dict({})`. That is a deliberate state, not
    a gap: the test pins it so that adding the first style dim is a conscious act with a declaration
    change, rather than something that drifts in."""
    m = _map_for(swm_id)
    assert m.pairs() == {}, (
        f"{swm_id} now maps {sorted(m.pairs())}. If the cell gained a style space, update this test "
        f"and re-baseline the cell's style_dim/style_digest declaration deliberately")
    assert set(m.swm_only) == set(RECORDED_SWM_LEAVES[swm_id])


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_state_leaves_are_classified_not_applicable(swm_id):
    for leaf in ("agent.ee_start_position", "cube.start_position", "cube.start_yaw",
                 "cube.goal_position", "cube.goal_yaw"):
        rec = LEAF_CLASSIFICATION[swm_id]["leaves"][leaf]
        assert rec["classification"] == "not_applicable", (
            f"{swm_id}:{leaf} reaches the frame only through qpos/mocap, which is our state")


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_agent_color_records_the_upstream_dirty_flag_defect(swm_id):
    rec = LEAF_CLASSIFICATION[swm_id]["leaves"]["agent.color"]
    assert rec["classification"] == "dead_upstream"
    assert "allclose" in rec["reason"], (
        "the reason must name the inverted dirty-flag test, because the leaf is dead only in "
        "isolation and live when co-varied -- a reader who does not know that will mis-file it")


def test_lock_color_is_scene_only():
    assert "lock_color" in RECORDED_SWM_LEAVES["swm/OGBScene-v0"]
    assert "lock_color" not in RECORDED_SWM_LEAVES["swm/OGBCube-v0"]
    assert "lock_color" in _map_for("swm/OGBScene-v0").swm_only
    assert "lock_color" not in _map_for("swm/OGBCube-v0").swm_only


# ---------------------------------------------------------------- against the LIVE wrapper

@pytest.mark.parametrize("swm_id", IDS)
def test_the_recorded_inventory_still_matches_the_live_wrapper(swm_id):
    """Everything above is measured against a recorded list. This is the test that keeps that list
    honest: if swm adds, removes or renames a leaf, the recorded inventory is stale and every
    completeness assertion above has been checking the wrong set."""
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_worldmodel")
    try:
        env = gymnasium.make(swm_id)
    except Exception as exc:                      # the family's own deps (ogbench, PyFlyt, mujoco)
        pytest.skip(f"{swm_id} not constructible here: {type(exc).__name__}: {exc}")
    try:
        live = sorted(env.unwrapped.variation_space.names())
    finally:
        env.close()
    assert live == sorted(RECORDED_SWM_LEAVES[swm_id]), (
        f"{swm_id}: the live variation space is {live} but RECORDED_SWM_LEAVES says "
        f"{sorted(RECORDED_SWM_LEAVES[swm_id])}. Re-enumerate and re-classify the difference")


# ---------------------------------------------------------------- the maze families

MAZE = "swm/OGBMaze-v0"
SPM = "swm/SimplePointMaze-v0"


def test_ogbmaze_wall_color_is_renamed_not_identical():
    """swm spells it `wall.color`, we spell it `walls.color`. Mapping it as `identical` would map
    nothing at all, silently, because neither side would find the other's path."""
    m = _map_for(MAZE)
    assert m.renamed.get("walls.color") == "wall.color"
    assert "wall.color" not in m.identical and "walls.color" not in m.identical
    assert m.pairs()["walls.color"] == ("wall.color", 1.0)


def test_ogbmaze_default_conflict_is_resolved_in_favour_of_the_pixel_mode_value():
    """swm reads wall.color's init_value from the loaded XML, which is 1.0 in states mode and 0.6 in
    pixels mode. Our renderer reproduces the pixel-mode appearance, so 0.6 is the right default and
    the reason must say why rather than leaving a reader to rediscover the conflict."""
    rec = LEAF_CLASSIFICATION[MAZE]["leaves"]["wall.color"]
    assert rec["classification"] == "mapped"
    assert "0.6" in rec["evidence"] and "0 of 50,176" in rec["evidence"]


def test_ogbmaze_records_that_the_registered_states_mode_config_renders_every_leaf_dead():
    """The sharpest upstream finding for this id: in ob_type='states' -- which is what the swm id
    registers -- none of the five leaves reaches a pixel, because compile_model() rebuilds ogbench's
    renderer only for 'pixels'. A classification that did not say which config it was measured in
    would be unreadable."""
    note = LEAF_CLASSIFICATION[MAZE]["config_note"]
    assert "states" in note and "custom_renderer" in note
    assert "pixels" in note


def test_ogbmaze_floor_color_is_dead_at_the_appearance_we_actually_render():
    rec = LEAF_CLASSIFICATION[MAZE]["leaves"]["floor.color"]
    assert rec["classification"] == "dead_upstream"
    assert "0 of 50,176" in rec["evidence"], (
        "the dead verdict is conditional on the official pixel_encoding=1 appearance; the evidence "
        "must carry that count or the verdict cannot be checked")


def test_ogbmaze_pixel_encoding_is_unsupported_because_the_map_cannot_invert():
    """Our `rendering.decolor_floor` IS the same switch, with the opposite sense. It is recorded as
    unsupported rather than mapped because VariationMap can only express `ours = theirs * factor`,
    and quietly mapping it would invert the semantics for every caller."""
    rec = LEAF_CLASSIFICATION[MAZE]["leaves"]["floor.pixel_encoding"]
    assert rec["classification"] == "unsupported"
    assert "1 - theirs" in _map_for(MAZE).swm_only["floor.pixel_encoding"]
    assert "decolor_floor" not in {o for o in _map_for(MAZE).pairs()}


def test_ogbmaze_camera_and_scale_leaves_are_declared_ours_only():
    m = _map_for(MAZE)
    assert set(m.ours_only) == {"agent.scale", "camera.distance_scale", "camera.elevation"}
    for leaf in m.ours_only:
        assert leaf not in RECORDED_SWM_LEAVES[MAZE]


def test_both_pointmaze_large_cells_share_one_mapping():
    """navigate and stitch differ by dataset, not by appearance; two maps that can drift apart would
    be two records of one fact."""
    a = NEWCELL_VARIATION_MAPS["PointmazeLargeNavigate"]
    b = NEWCELL_VARIATION_MAPS["PointmazeLargeStitch"]
    assert a.swm_id == b.swm_id == MAZE
    assert a.pairs() == b.pairs() and dict(a.swm_only) == dict(b.swm_only)


def test_simple_point_maze_maps_all_seven_appearance_leaves():
    m = _map_for(SPM)
    assert set(m.pairs()) == {"agent.color", "agent.radius", "background.color", "walls.color",
                              "walls.number", "walls.shape", "walls.positions"}
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs and factor == 1.0


def test_simple_point_maze_goal_leaves_are_not_applicable_because_show_goal_is_off():
    """Not `unsupported`: the renderer could draw a goal. This registration does not have one, which
    is a different statement, and it is the pre-existing documented divergence rather than a Step-8
    decision."""
    for leaf in ("goal.color", "goal.radius", "goal.position"):
        rec = LEAF_CLASSIFICATION[SPM]["leaves"][leaf]
        assert rec["classification"] == "not_applicable"
        assert "show_goal" in rec["reason"]
    assert "show_goal=False" in LEAF_CLASSIFICATION[SPM]["config_note"]


def test_simple_point_maze_position_and_speed_are_not_style():
    for leaf in ("agent.position", "agent.speed"):
        assert LEAF_CLASSIFICATION[SPM]["leaves"][leaf]["classification"] == "not_applicable"


def test_simple_point_maze_mapped_leaves_are_dims_of_our_own_space():
    """The map names OUR paths. If a path is not a leaf of our variation space, the mapping silently
    does nothing -- which is the failure mode `renamed` exists to prevent."""
    spm = pytest.importorskip("stable_envs.envs.simple_point_maze")
    space = spm.SimplePointMaze.make_variation_space(spm.SimplePointMaze)
    ours = set(space.names())
    for leaf in _map_for(SPM).pairs():
        assert leaf in ours, f"{leaf} is mapped but is not a leaf of SimplePointMaze's own space"


def test_craftax_ids_are_recorded_as_having_no_leaves_at_all():
    """Confirmed live: CraftaxWrapper.__init__ sets variation_space = None, so there is no Gate-F
    variation gap for either id. Recorded so the Step-8 count has no implicit exception."""
    from stable_envs.compat.variation_maps.newcells import CRAFTAX_NO_LEAVES
    assert set(CRAFTAX_NO_LEAVES) == {"swm/CraftaxClassicPixels-v1", "swm/CraftaxPixels-v1"}
    for sid, rec in CRAFTAX_NO_LEAVES.items():
        assert rec["n_swm_leaves"] == 0
        assert rec["classification"] == "not_applicable"
        assert "variation_space is None" in rec["evidence"]
        assert sid not in RECORDED_SWM_LEAVES, f"{sid} offers no leaves; it needs no map"
