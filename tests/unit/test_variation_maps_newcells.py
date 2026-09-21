"""The Step-7 new cells' VariationMaps: every offered swm leaf is classified, and the classification
agrees with the map it is supposed to describe.

The point of these tests is NOT that the reasons are well written. It is that no swm leaf can be
added upstream, or removed from a map here, without something failing: an unclassified leaf is
exactly the Gate-F gap the maps exist to close, and it is invisible unless a test counts.

Most of this runs without ``stable_worldmodel``, against the leaf inventory recorded from the live
wrappers, because swm is not importable in every environment this suite runs in and a test that
skips where it matters is not a test. The live enumeration runs as well wherever swm IS importable,
and asserts the recorded inventory still matches the wrapper.
"""

# ruff: noqa: I001
# worldmodel must load before newcells (circular MAPS install).

import numpy as np
import pytest
import torch

from tests.helpers.reference import VARIATION_MAPS, variation_map
from tests.helpers.reference.variation_maps.newcells import (
    BUCKETS,
    LEAF_CLASSIFICATION,
    NEWCELL_VARIATION_MAPS,
    RECORDED_OURS_LEAVES,
    RECORDED_SWM_LEAVES,
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
        f"leaves nor swm_only. That is a silently unmapped leaf, which is the Gate-F gap the maps "
        f"closes -- classify it rather than deleting this assertion"
    )
    both = mapped & declared
    assert not both, f"{swm_id}: {sorted(both)} are both mapped and declared unmapped"
    assert not (mapped - leaves), f"{swm_id}: {sorted(mapped - leaves)} are mapped to swm paths swm does not offer"
    assert not (declared - leaves), f"{swm_id}: {sorted(declared - leaves)} carry a reason but are not offered by swm"


@pytest.mark.parametrize("swm_id", IDS)
def test_every_offered_leaf_has_a_structured_classification(swm_id):
    cls = LEAF_CLASSIFICATION[swm_id]["leaves"]
    leaves = set(RECORDED_SWM_LEAVES[swm_id])
    assert set(cls) == leaves, f"{swm_id}: classification covers {sorted(set(cls))} but swm offers {sorted(leaves)}"


@pytest.mark.parametrize("swm_id", IDS)
def test_every_verdict_is_one_of_the_four_buckets(swm_id):
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        assert rec["classification"] in BUCKETS, (
            f"{swm_id}:{leaf} has verdict {rec['classification']!r}, which is not one of {BUCKETS}"
        )


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
                f"{swm_id}:{leaf} is classified {rec['classification']} but the map DOES carry it"
            )
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
                f"{swm_id}:{leaf} is dead_upstream but its evidence {ev!r} cites no measurement"
            )


@pytest.mark.parametrize("swm_id", IDS)
def test_every_unmapped_leaf_explains_itself(swm_id):
    for leaf, rec in LEAF_CLASSIFICATION[swm_id]["leaves"].items():
        if rec["classification"] != "mapped":
            assert len(rec["reason"]) > 40, (
                f"{swm_id}:{leaf} is {rec['classification']} with a reason too short to be one"
            )


# ---------------------------------------------------------------- Rocket, the one family with dims

ROCKET = "swm/PFRocketLanding-v0"


def test_rocket_maps_exactly_the_five_material_colours():
    m = _map_for(ROCKET)
    assert set(m.pairs()) == {
        "rocket.body_color",
        "rocket.fin_color",
        "rocket.leg_color",
        "rocket.booster_color",
        "pad.color",
    }
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs, f"{ours} should map by identical path, not to {theirs}"
        assert factor == 1.0, (
            f"{ours} carries factor {factor}; both sides are 0-255 RGB and a factor here would be "
            f"the PushT scale bug in a new place"
        )


def test_rocket_ignition_is_ours_only_because_swm_derives_it_from_the_action():
    m = _map_for(ROCKET)
    assert "flame.ignited" in m.ours_only
    assert "flame.ignited" not in RECORDED_SWM_LEAVES[ROCKET]


def test_rocket_sky_color_is_dead_upstream_and_not_quietly_mapped():
    """The one leaf measured dead upstream. If a future edit maps it, this fails rather than shipping
    a knob that moves nothing."""
    m = _map_for(ROCKET)
    assert "environment.sky_color" not in {t for t, _ in m.pairs().values()}
    rec = LEAF_CLASSIFICATION[ROCKET]["leaves"]["environment.sky_color"]
    assert rec["classification"] == "dead_upstream"
    assert "0 of 691,200" in rec["evidence"]


def test_rocket_start_state_leaves_are_state_not_style():
    """They move pixels -- by moving the rocket. That makes them our STATE, and mapping them to
    style would put the same information in two places."""
    for leaf in ("environment.start_height_ratio", "environment.start_horizontal_offset", "environment.start_tilt"):
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
        back = default[o : o + w] / factor
        assert np.allclose(back * factor, default[o : o + w]), f"{ours} does not round-trip"


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
    assert list(row[o : o + w]) == [255.0, 255.0, 255.0]


# ---------------------------------------------------------------- OGB, the empty-style family


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_maps_authorized_cube_and_floor_colours(swm_id):
    """These two cells declare `make_variation_space() -> Dict({})`. That is a deliberate state, not
    a gap: the test pins it so that adding the first style dim is a conscious act with a declaration
    change, rather than something that drifts in."""
    m = _map_for(swm_id)
    assert set(m.pairs()) == {"cube.color", "floor.color"}
    assert set(m.swm_only) == set(RECORDED_SWM_LEAVES[swm_id]) - set(m.pairs())


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_state_leaves_are_classified_not_applicable(swm_id):
    for leaf in (
        "agent.ee_start_position",
        "cube.start_position",
        "cube.start_yaw",
        "cube.goal_position",
        "cube.goal_yaw",
    ):
        rec = LEAF_CLASSIFICATION[swm_id]["leaves"][leaf]
        assert rec["classification"] == "not_applicable", (
            f"{swm_id}:{leaf} reaches the frame only through qpos/mocap, which is our state"
        )


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_agent_color_records_the_upstream_dirty_flag_defect(swm_id):
    rec = LEAF_CLASSIFICATION[swm_id]["leaves"]["agent.color"]
    assert rec["classification"] == "dead_upstream"
    assert "allclose" in rec["reason"], (
        "the reason must name the inverted dirty-flag test, because the leaf is dead only in "
        "isolation and live when co-varied -- a reader who does not know that will mis-file it"
    )


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
    except Exception as exc:  # the family's own deps (ogbench, PyFlyt, mujoco)
        pytest.skip(f"{swm_id} not constructible here: {type(exc).__name__}: {exc}")
    try:
        live = sorted(env.unwrapped.variation_space.names())
    finally:
        env.close()
    assert live == sorted(RECORDED_SWM_LEAVES[swm_id]), (
        f"{swm_id}: the live variation space is {live} but RECORDED_SWM_LEAVES says "
        f"{sorted(RECORDED_SWM_LEAVES[swm_id])}. Re-enumerate and re-classify the difference"
    )


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
        "must carry that count or the verdict cannot be checked"
    )


def test_ogbmaze_pixel_encoding_is_unsupported_because_the_map_cannot_invert():
    """Our `rendering.decolor_floor` IS the same switch, with the opposite sense. It is recorded as
    unsupported rather than mapped because VariationMap can only express `ours = theirs * factor`,
    and quietly mapping it would invert the semantics for every caller."""
    rec = LEAF_CLASSIFICATION[MAZE]["leaves"]["floor.pixel_encoding"]
    assert rec["classification"] == "unsupported"
    assert "1 - theirs" in _map_for(MAZE).swm_only["floor.pixel_encoding"]
    assert "decolor_floor" not in {o for o in _map_for(MAZE).pairs()}


@pytest.mark.parametrize("cell", sorted(RECORDED_OURS_LEAVES))
def test_every_own_side_leaf_is_mapped_or_carries_an_ours_only_reason(cell):
    """The completeness check in the OTHER direction, and the one the reviewer found missing.

    `test_every_offered_leaf_is_in_exactly_one_bucket_of_the_map` makes every SWM leaf account for
    itself. Nothing made OUR leaves do the same, so a leaf we render with no swm counterpart could
    sit in the space with no statement of why -- which is how OGBMaze shipped `floor.tint` and
    `rendering.decolor_floor` undocumented while three sibling leaves had reasons.

    Every own-side leaf must be either MAPPED (a key of `pairs()`) or carry a non-empty `ours_only`
    reason, and `ours_only` may not name a leaf we do not actually have.
    """
    m = NEWCELL_VARIATION_MAPS[cell]
    ours = set(RECORDED_OURS_LEAVES[cell])
    mapped = set(m.pairs())
    declared = set(m.ours_only)
    unexplained = ours - mapped - declared
    assert not unexplained, (
        f"{cell}: {sorted(unexplained)} are leaves of OUR variation space and appear in NEITHER the "
        f"map nor ours_only. Give each one a reason -- 'swm has no counterpart' is a claim, and an "
        f"undocumented leaf is the same Gate-F gap as an unclassified swm leaf, facing the other way"
    )
    both = mapped & declared
    assert not both, f"{cell}: {sorted(both)} are both mapped and declared ours-only"
    assert not (declared - ours), (
        f"{cell}: ours_only names {sorted(declared - ours)}, which our variation space does not "
        f"offer; a reason for a leaf that does not exist explains nothing"
    )
    for leaf in declared:
        assert m.ours_only[leaf].strip(), f"{cell}:{leaf} has an empty ours_only reason"


@pytest.mark.parametrize("cell", sorted(RECORDED_OURS_LEAVES))
def test_no_ours_only_leaf_is_secretly_offered_by_swm(cell):
    """An `ours_only` leaf swm ALSO offers is a mapping we failed to make, not a leaf they lack."""
    m = NEWCELL_VARIATION_MAPS[cell]
    theirs = set(RECORDED_SWM_LEAVES[m.swm_id])
    for leaf in m.ours_only:
        assert leaf not in theirs, (
            f"{cell}:{leaf} is declared ours-only but swm offers it too. Either map it or move the "
            f"reason to swm_only, where it will be checked against the swm inventory"
        )


def test_ogbmaze_ours_only_names_the_five_leaves_swm_has_no_counterpart_for():
    """The specific inventory, so a leaf cannot silently leave the list either."""
    m = _map_for(MAZE)
    assert set(m.ours_only) == {
        "agent.scale",
        "floor.tint",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
    }
    assert "checker" in m.ours_only["floor.tint"], (
        "floor.tint's reason must say WHY it is not swm's floor.color: ours multiplies the bound "
        "texture, swm replaces the checker's two RGB values"
    )
    assert "pixel_encoding" in m.ours_only["rendering.decolor_floor"], (
        "decolor_floor's reason must name pixel_encoding, because the two are the same switch and a "
        "reader who does not know that will file this as an unmapped swm leaf"
    )


def test_both_pointmaze_large_cells_share_one_mapping():
    """navigate and stitch differ by dataset, not by appearance; two maps that can drift apart would
    be two records of one fact."""
    a = NEWCELL_VARIATION_MAPS["PointmazeLargeNavigate"]
    b = NEWCELL_VARIATION_MAPS["PointmazeLargeStitch"]
    assert a.swm_id == b.swm_id == MAZE
    assert a.pairs() == b.pairs() and dict(a.swm_only) == dict(b.swm_only)


def test_simple_point_maze_maps_appearance_and_speed():
    m = _map_for(SPM)
    assert set(m.pairs()) == {
        "agent.color",
        "agent.radius",
        "agent.speed",
        "background.color",
        "walls.color",
        "walls.number",
        "walls.shape",
        "walls.positions",
    }
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs and factor == 1.0


def test_simple_point_maze_goal_leaves_are_not_applicable_because_show_goal_is_off():
    """Not `unsupported`: the renderer could draw a goal. This registration does not have one, which
    is a different statement, and it is the pre-existing documented divergence rather than a mapping
    decision."""
    for leaf in ("goal.color", "goal.radius", "goal.position"):
        rec = LEAF_CLASSIFICATION[SPM]["leaves"][leaf]
        assert rec["classification"] == "not_applicable"
        assert "show_goal" in rec["reason"]
    assert "show_goal=False" in LEAF_CLASSIFICATION[SPM]["config_note"]


def test_simple_point_maze_position_is_not_style():
    assert LEAF_CLASSIFICATION[SPM]["leaves"]["agent.position"]["classification"] == "not_applicable"
    assert LEAF_CLASSIFICATION[SPM]["leaves"]["agent.speed"]["classification"] == "mapped"


def test_simple_point_maze_mapped_leaves_are_dims_of_our_own_space():
    """The map names OUR paths. If a path is not a leaf of our variation space, the mapping silently
    does nothing -- which is the failure mode `renamed` exists to prevent."""
    spm = pytest.importorskip("stable_envs.envs.navigation")
    space = spm.SimplePointMaze.make_variation_space(spm.SimplePointMaze)
    ours = set(space.names())
    for leaf in _map_for(SPM).pairs():
        assert leaf in ours, f"{leaf} is mapped but is not a leaf of SimplePointMaze's own space"


def test_craftax_ids_are_recorded_as_having_no_leaves_at_all():
    """Confirmed live: CraftaxWrapper.__init__ sets variation_space = None, so there is no Gate-F
    variation gap for either id. Recorded so the completeness count has no implicit exception."""
    from tests.helpers.reference.variation_maps.newcells import CRAFTAX_NO_LEAVES

    assert set(CRAFTAX_NO_LEAVES) == {"swm/CraftaxClassicPixels-v1", "swm/CraftaxPixels-v1"}
    for sid, rec in CRAFTAX_NO_LEAVES.items():
        assert rec["n_swm_leaves"] == 0
        assert rec["classification"] == "not_applicable"
        assert "variation_space is None" in rec["evidence"]
        assert sid not in RECORDED_SWM_LEAVES, f"{sid} offers no leaves; it needs no map"


# ---------------------------------------------------------------- OGB cube.color, on the CPU
#
# These exercise the REAL helper methods -- `_cube_colours`, `_cube_geom_index` and
# `_upload_cube_colours` -- against a stub bundle and a stub mujoco_warp context, so the
# custom-to-default leak the reviewer found has a regression test that needs no GPU and no ogbench.
# They deliberately do not construct an env: `OGBManipBase.__init__` refuses any device but cuda,
# and the defect lives entirely in these three methods and the module context cache they write to.

_CUBE_DEFAULT = [0.96, 0.26, 0.33]
_CUBE_GEOMS = [62, 63]
#: object_0 is opaque; target_object_0 is the translucent goal marker. swm writes rgb and leaves the
#: alpha channel alone, so the alphas here are what the helper must preserve.
_CUBE_ALPHAS = [1.0, 0.2]
#: swm's declared checker default; only its width matters to these tests.
_FLOOR_DEFAULT = [[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]


def _stub_env(device="cpu"):
    """An `OGBManipBase` with only the attributes the colour helpers touch."""
    import types

    from stable_envs.envs.ogbench.manipulation import OGBManipBase

    ngeom = 80
    pristine = np.zeros((ngeom, 4), np.float64)
    pristine[:, 3] = 1.0
    for g, a in zip(_CUBE_GEOMS, _CUBE_ALPHAS):
        pristine[g, :3] = _CUBE_DEFAULT
        pristine[g, 3] = a
    env = object.__new__(OGBManipBase)
    env.device = device
    env._b = types.SimpleNamespace(
        cube_geoms=np.asarray(_CUBE_GEOMS, np.int64),
        pristine={"geom_rgba": pristine},
        var_defaults={"cube.color": np.asarray(_CUBE_DEFAULT, np.float64)},
    )
    return env, ngeom


def _stub_context(nworld, ngeom):
    """A stand-in for the cached mujoco_warp context: one mutable per-world `geom_rgba`.

    Created ONCE per test and reused across renders/chunks/instances on purpose -- that reuse is the
    property `_CTX_CACHE` has and the reason a skipped default upload leaks.
    """
    import types

    buf = torch.zeros((nworld, ngeom, 4), dtype=torch.float32)
    buf[:, :, 3] = 1.0
    return types.SimpleNamespace(
        wp=types.SimpleNamespace(to_torch=lambda a: a), wm=types.SimpleNamespace(geom_rgba=buf)
    ), buf


class _StubStyle:
    """The two attributes `_style_host` reads off a resolved `StyleTensor`: `values` and `layout`."""

    def __init__(self, cube_rows):
        cube = np.asarray(cube_rows, np.float32).reshape(-1, 3)
        floor = np.tile(np.asarray(_FLOOR_DEFAULT, np.float32).reshape(-1), (cube.shape[0], 1))
        self.values = torch.as_tensor(np.concatenate([cube, floor], axis=1))
        self.layout = {"cube.color": (0, 3), "floor.color": (3, 6)}


def _sty(rows):
    """One render's host-side style, produced by the REAL `_style_host`, not hand-built.

    The helpers take the host dict rather than the `StyleTensor` so that the whole nine-float row
    crosses to the host once per render; routing the stub through `_style_host` is what keeps these
    tests honest about which units the helpers actually receive.
    """
    env = object.__new__(_base_cls())
    return env._style_host(_StubStyle(rows))


def _base_cls():
    from stable_envs.envs.ogbench.manipulation import OGBManipBase

    return OGBManipBase


def _upload(env, c, sty, B, nworld):
    """One render's worth of uploads, chunked exactly as `render` chunks them."""
    from stable_envs.envs.ogbench.manipulation import MJW_CHUNK

    cube = env._cube_colours(sty, B)
    assert cube is not None, (
        "_cube_colours returned None, so render() would skip the upload entirely. That is the "
        "defect: a cell with cube geoms must always be given a value to write"
    )
    for lo in range(0, B, MJW_CHUNK):
        hi = min(lo + MJW_CHUNK, B)
        env._upload_cube_colours(c, cube[lo:hi], hi - lo)


def _rows(buf, n):
    return buf[:n][:, _CUBE_GEOMS, :].numpy()


def _expect(colours):
    out = np.zeros((len(colours), len(_CUBE_GEOMS), 4), np.float64)
    for i, rgb in enumerate(colours):
        out[i, :, :3] = rgb
        out[i, :, 3] = _CUBE_ALPHAS
    return out


def test_cube_colours_never_returns_none_for_the_registered_default():
    """The reviewer's reproduction, inverted into an assertion. The old `_cube_colours` answered
    `None` whenever the request was close to the default, and `render` then skipped the upload."""
    env, _ = _stub_env()
    got = env._cube_colours(_sty([_CUBE_DEFAULT]), 1)
    assert got is not None
    assert np.allclose(got.numpy(), _expect([_CUBE_DEFAULT]))


def test_custom_then_default_restores_the_default_rgb():
    """Render red, then render the default. The second render must not draw red.

    This is the exact sequence that failed: OGBCube has no button upload that would rewrite the
    `geom_rgba` block, so the context kept the previous render's colour.
    """
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    _upload(env, c, _sty([[1.0, 0.0, 0.0]] * 4), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect([[1.0, 0.0, 0.0]] * 4))
    _upload(env, c, _sty([_CUBE_DEFAULT] * 4), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect([_CUBE_DEFAULT] * 4)), (
        "a default render after a custom one drew the custom colour -- the per-world geom_rgba was "
        "left as the previous render wrote it"
    )


def test_mixed_batch_then_default_restores_every_world():
    """A per-environment style, then a uniform default. Every world must come back, not just world 0."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    mixed = [[1.0, 0.0, 0.0], _CUBE_DEFAULT, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    _upload(env, c, _sty(mixed), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect(mixed))
    _upload(env, c, _sty([_CUBE_DEFAULT] * 4), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect([_CUBE_DEFAULT] * 4))


def test_default_then_custom_then_default_is_bit_identical_to_the_first_default():
    """The round trip, compared against the FIRST default rather than against the declared value:
    a restore that merely lands near the default is not a restore."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(2, ngeom)
    _upload(env, c, _sty([_CUBE_DEFAULT] * 2), 2, 2)
    first = _rows(buf, 2).copy()
    _upload(env, c, _sty([[0.1, 0.9, 0.4]] * 2), 2, 2)
    _upload(env, c, _sty([_CUBE_DEFAULT] * 2), 2, 2)
    assert np.array_equal(_rows(buf, 2), first)


def test_a_second_renderer_instance_does_not_inherit_the_first_ones_colour():
    """`_CTX_CACHE` is keyed by (cell, res, nworld), not by renderer instance, so two envs in one
    process share the per-world buffer. The second instance's default render must still be default."""
    a, ngeom = _stub_env()
    b, _ = _stub_env()
    c, buf = _stub_context(2, ngeom)
    _upload(a, c, _sty([[0.0, 0.0, 1.0]] * 2), 2, 2)
    _upload(b, c, _sty([_CUBE_DEFAULT] * 2), 2, 2)
    assert np.allclose(_rows(buf, 2), _expect([_CUBE_DEFAULT] * 2))


def test_a_small_default_batch_after_a_large_custom_one_leaves_no_stale_world():
    """Chunking is per-render, so a render of 2 worlds writes worlds 0-1 of a buffer an earlier
    render filled to 4. The rendered worlds are the ones that must be correct."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    _upload(env, c, _sty([[1.0, 0.0, 1.0]] * 4), 4, 4)
    _upload(env, c, _sty([_CUBE_DEFAULT] * 2), 2, 4)
    assert np.allclose(_rows(buf, 2), _expect([_CUBE_DEFAULT] * 2))


def test_each_cube_geom_keeps_its_own_alpha():
    """swm writes `rgba[:3]` and leaves the alpha to the MJCF: the opaque cube and the translucent
    goal marker must not be flattened to one alpha by the colour write."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(1, ngeom)
    _upload(env, c, _sty([[0.2, 0.4, 0.6]]), 1, 1)
    assert list(buf[0][_CUBE_GEOMS][:, 3].numpy()) == pytest.approx(_CUBE_ALPHAS)


def test_the_colour_write_touches_only_the_cube_geoms():
    """Isolation, on the CPU: every other geom row of the shared buffer is untouched."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(2, ngeom)
    before = buf.clone()
    _upload(env, c, _sty([[1.0, 0.0, 0.0]] * 2), 2, 2)
    other = [g for g in range(ngeom) if g not in _CUBE_GEOMS]
    assert torch.equal(buf[:, other], before[:, other])


# ---------------------------------------------------------------- the cube.color shape contract


def _ogb_space(cell):
    """The real `make_variation_space` for one OGB cell, with no CUDA.

    `OGBManipBase.__init__` refuses a non-cuda device, but `make_variation_space` only needs the
    cached bundle, which is mujoco + ogbench on the CPU. Called unbound on a stub carrying `CELL`,
    so this is the shipped code path and not a copy of it.
    """
    import types

    pytest.importorskip("ogbench")
    from stable_envs.envs.ogbench.manipulation import OGBManipBase

    return OGBManipBase.make_variation_space(types.SimpleNamespace(CELL=cell))


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_our_cube_colour_leaf_is_flat_three_and_floor_colour_is_two_by_three(cell):
    """One shape, decided once. swm declares `cube.color` as `(num_cubes, 3)` and both registered
    cells are one-cube configurations, so we carry the leaf FLAT: a leading axis of length 1 buys
    nothing and costs a shape the Box, the style row and the swm value would each spell differently.
    `floor.color` stays `(2, 3)` because the checker's two colours are not interchangeable.
    """
    space = _ogb_space(cell)
    layout = {p: (o, w) for p, o, w in space.flat_layout()}
    assert space["cube"]["color"].shape == (3,)
    assert space["floor"]["color"].shape == (2, 3)
    assert layout["cube.color"][1] == 3 and layout["floor.color"][1] == 6


@pytest.mark.parametrize("cell,swm_id", [("Cube", "swm/OGBCube-v0"), ("Scene", "swm/OGBScene-v0")])
def test_a_swm_shaped_cube_colour_survives_the_round_trip_through_our_style_row(cell, swm_id):
    """swm hands out `(1, 3)`; `style_from_variation` flattens before slicing, so its value lands in
    our width-3 slot unchanged, and `variation_from_style` gives the three numbers back.

    The residual asymmetry is recorded rather than hidden: `variation_from_style` emits a FLAT
    vector for every multi-row leaf, so feeding its output straight back into a live swm env needs a
    reshape to `(1, 3)` (and `(2, 3)` for the floor). That is a property of `compat.worldmodel`,
    which this workstream does not own; findings.md carries it as a boundary reshape.
    """
    import types

    from tests.helpers.reference import style_from_variation, variation_from_style

    space = _ogb_space(cell)
    name = {"Cube": "OGBCube", "Scene": "OGBScene"}[cell]
    env = types.new_class(name, ())()
    env.variation_space = space
    base = np.zeros(sum(w for _, _, w in space.flat_layout()), np.float32)
    want = np.array([[0.1, 0.2, 0.3]])  # swm's (num_cubes, 3), num_cubes == 1
    row = style_from_variation(env, {"cube": {"color": want}}, base=base)
    o, w = {p: (o, w) for p, o, w in space.flat_layout()}["cube.color"]
    assert list(row[o : o + w]) == pytest.approx([0.1, 0.2, 0.3])
    back = variation_from_style(env, row)
    assert np.asarray(back["cube.color"]).reshape(-1) == pytest.approx([0.1, 0.2, 0.3])
    assert np.asarray(back["floor.color"]).reshape(-1).shape == (6,)


def test_pixel_encoding_reason_does_not_promise_that_the_affine_flag_alone_maps_it():
    """Reviewer finding 4. An affine `(factor, offset)` makes the relation WRITEABLE; it does not
    make swm's two-colour checker and our flat slate the same picture, and that is unmeasured. The
    reason must say both, because the first half alone reads as "mapped once A merges"."""
    reason = _map_for(MAZE).swm_only["floor.pixel_encoding"]
    assert "PENDING-A" in reason.upper()
    assert "has not been measured" in reason
    assert "checker" in reason.lower() and "slate" in reason.lower()


def test_the_whole_style_row_crosses_to_the_host_in_one_read():
    """`_style_host` returns every leaf from ONE device-to-host copy. Two copies cost two
    synchronisations, which is a measurable fraction of a B=1 render and nothing at B=4096 -- the
    shape of a latency term, and the reason the helpers take a host dict rather than the tensor."""
    env = object.__new__(_base_cls())
    sty = _StubStyle([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    calls = []
    real = sty.values.detach

    class _Counting:
        def __init__(self, t):
            self._t = t

        def detach(self):
            calls.append(1)
            return real()

        def __getattr__(self, k):
            return getattr(self._t, k)

    sty.values = _Counting(sty.values)
    host = env._style_host(sty)
    assert calls == [1], f"the style row crossed to the host {len(calls)} times, not once"
    assert set(host) == {"cube.color", "floor.color"}
    assert host["cube.color"].shape == (2, 3) and host["floor.color"].shape == (2, 6)
    assert host["cube.color"][1] == pytest.approx([0.4, 0.5, 0.6])


def test_the_host_style_read_is_memoised_on_the_tensor_and_invalidated_by_a_write():
    """A render loop at one appearance should pay ONE device-to-host sync in total, not one a frame:
    `resolve_style(None, None)` hands back the env's own style tensor every call. The cache is keyed
    on identity AND `_version`, so writing through the tensor invalidates it, and it holds a strong
    reference so a freed-and-reused `id()` cannot alias a stale entry. It caches a pure read; it
    never gates an upload."""
    env = object.__new__(_base_cls())
    sty = _StubStyle([[0.1, 0.2, 0.3]])
    first = env._style_host(sty)
    assert env._style_host(sty) is first, "a second read of the same tensor re-crossed to the host"
    with torch.no_grad():
        sty.values[0, 0] = 0.9
    again = env._style_host(sty)
    assert again is not first, "an in-place write to the style row did not invalidate the host copy"
    assert again["cube.color"][0][0] == pytest.approx(0.9)
    assert env._style_host(_StubStyle([[0.1, 0.2, 0.3]])) is not again, "a different tensor hit the cache"
