"""The OGB / Rocket / maze VariationMaps: every offered swm leaf is classified, and the classification
agrees with the map.

Most checks run against the leaf inventory recorded from the live wrappers, so they need no
``stable_worldmodel``. Where swm is importable, a live test asserts the recorded inventory still
matches the wrapper.
"""

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
    ROCKET_START_LEAVES,
)

IDS = sorted(RECORDED_SWM_LEAVES)


def _map_for(swm_id):
    ms = [m for m in NEWCELL_VARIATION_MAPS.values() if m.swm_id == swm_id]
    assert ms, f"no VariationMap declares swm_id={swm_id}"
    return ms[0]


# ---------------------------------------------------------------- registration


def test_the_new_maps_are_registered_in_the_shared_table():
    """``tests/helpers/reference/variation.py`` installs these into ``VARIATION_MAPS``."""
    for name, m in NEWCELL_VARIATION_MAPS.items():
        assert VARIATION_MAPS.get(name) is m, f"{name} is not registered in VARIATION_MAPS"
        assert variation_map(name) is m


def test_registering_the_new_maps_did_not_disturb_the_pre_existing_ones():
    for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "PushT"):
        assert name in VARIATION_MAPS, f"{name} is missing from VARIATION_MAPS"


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
        f"{swm_id}: {sorted(undocumented)} are offered by swm and appear in neither the mapped leaves nor swm_only"
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
    """`mapped` in the classification table means mapped in the VariationMap, and vice versa."""
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
    """A `dead_upstream` verdict cites a measurement of the official renderer (pixels or arrays)."""
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


ROCKET_COLOURS = {"rocket.body_color", "rocket.fin_color", "rocket.leg_color", "rocket.booster_color", "pad.color"}


def test_rocket_maps_the_five_material_colours_and_the_start_leaves():
    m = _map_for(ROCKET)
    assert set(m.pairs()) == ROCKET_COLOURS | set(ROCKET_START_LEAVES)
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs, f"{ours} should map by identical path, not to {theirs}"
        assert factor == 1.0, f"{ours} carries factor {factor}; both sides are 0-255 RGB"


def test_rocket_ignition_is_ours_only_because_swm_derives_it_from_the_action():
    m = _map_for(ROCKET)
    assert "flame.ignited" in m.ours_only
    assert "flame.ignited" not in RECORDED_SWM_LEAVES[ROCKET]


def test_rocket_sky_color_is_dead_upstream_and_not_quietly_mapped():
    """The one leaf measured dead upstream must stay unmapped."""
    m = _map_for(ROCKET)
    assert "environment.sky_color" not in {t for t, _ in m.pairs().values()}
    rec = LEAF_CLASSIFICATION[ROCKET]["leaves"]["environment.sky_color"]
    assert rec["classification"] == "dead_upstream"
    assert "0 of 691,200" in rec["evidence"]


def test_rocket_start_leaves_are_mapped_variation_but_not_renderer_style():
    """They move pixels by moving the rocket, so the renderer reads the pose from State, never them."""
    rl = pytest.importorskip("stable_envs.envs.rocket.legacy")
    assert not set(ROCKET_START_LEAVES) & set(rl.style_layout())
    for leaf in ROCKET_START_LEAVES:
        assert LEAF_CLASSIFICATION[ROCKET]["leaves"][leaf]["classification"] == "mapped"


def test_rocket_default_variation_round_trips_through_the_style_layout():
    """The registered default survives `variation -> style -> variation` unchanged."""
    rl = pytest.importorskip("stable_envs.envs.rocket.legacy")
    layout = rl.style_layout()
    default = np.asarray(rl._default_style_row(), np.float64)
    m = _map_for(ROCKET)
    for ours, (theirs, factor) in m.pairs().items():
        if ours in ROCKET_START_LEAVES:
            continue
        assert ours in layout, f"{ours} is mapped but is not a dim of the env's style layout"
        o, w = layout[ours]
        back = default[o : o + w] / factor
        assert np.allclose(back * factor, default[o : o + w]), f"{ours} does not round-trip"


def test_rocket_mapped_leaves_are_disjoint_slices_of_the_style_row():
    """Mapped leaves occupy disjoint slices of the style row, so each varies independently."""
    rl = pytest.importorskip("stable_envs.envs.rocket.legacy")
    layout = rl.style_layout()
    m = _map_for(ROCKET)
    seen = {}
    for ours in set(m.pairs()) - set(ROCKET_START_LEAVES):
        o, w = layout[ours]
        for i in range(o, o + w):
            assert i not in seen, f"{ours} and {seen[i]} both own style dim {i}"
            seen[i] = ours


def test_rocket_pad_color_default_is_white_not_the_declared_two_hundred():
    """swm declares pad.color (200,200,200), but a plain reset loads the unmodified pad URDF
    (`_modify_pad_urdf` runs only on a variation request), so the default is white."""
    rl = pytest.importorskip("stable_envs.envs.rocket.legacy")
    layout = rl.style_layout()
    o, w = layout["pad.color"]
    row = np.asarray(rl._default_style_row(), np.float64)
    assert list(row[o : o + w]) == [255.0, 255.0, 255.0]


# ---------------------------------------------------------------- OGB, the empty-style family


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_maps_recovered_public_configuration(swm_id):
    """The public semantic adapter has a wider configuration than the frozen renderer."""
    m = _map_for(swm_id)
    cell = "OGBCube" if "Cube" in swm_id else "OGBScene"
    assert set(m.pairs()) == set(RECORDED_OURS_LEAVES[cell])
    assert set(m.pairs()) == set(RECORDED_SWM_LEAVES[swm_id])
    assert not m.swm_only


@pytest.mark.parametrize("swm_id", ["swm/OGBCube-v0", "swm/OGBScene-v0"])
def test_ogb_reset_configuration_scope_is_explicit(swm_id):
    for leaf in (
        "agent.ee_start_position",
        "cube.start_position",
        "cube.start_yaw",
        "cube.goal_position",
        "cube.goal_yaw",
    ):
        assert LEAF_CLASSIFICATION[swm_id]["leaves"][leaf]["classification"] == "mapped"


def test_lock_color_is_scene_only():
    assert "lock_color" in RECORDED_SWM_LEAVES["swm/OGBScene-v0"]
    assert "lock_color" not in RECORDED_SWM_LEAVES["swm/OGBCube-v0"]
    assert "lock_color" in _map_for("swm/OGBScene-v0").pairs()
    assert "lock_color" not in _map_for("swm/OGBCube-v0").swm_only


# ---------------------------------------------------------------- against the LIVE wrapper


@pytest.mark.swm
@pytest.mark.parametrize("swm_id", IDS)
def test_the_recorded_inventory_still_matches_the_live_wrapper(swm_id):
    """The recorded leaf inventory the tests above use still matches the live wrapper."""
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
    """swm spells it `wall.color` and we spell it `walls.color`, so it must be renamed, not identical."""
    m = _map_for(MAZE)
    assert m.renamed.get("walls.color") == "wall.color"
    assert "wall.color" not in m.identical and "walls.color" not in m.identical
    assert m.pairs()["walls.color"] == ("wall.color", 1.0)


def test_ogbmaze_default_conflict_is_resolved_in_favour_of_the_pixel_mode_value():
    """swm's wall.color default is 1.0 in states mode and 0.6 in pixels mode; ours is the pixel-mode 0.6."""
    rec = LEAF_CLASSIFICATION[MAZE]["leaves"]["wall.color"]
    assert rec["classification"] == "mapped"
    assert "0.6" in rec["evidence"] and "0 of 50,176" in rec["evidence"]


def test_ogbmaze_records_that_the_registered_states_mode_config_renders_every_leaf_dead():
    """The config note records that in the registered ob_type='states' no leaf reaches a pixel.

    ``compile_model()`` rebuilds ogbench's renderer only for 'pixels'.
    """
    note = LEAF_CLASSIFICATION[MAZE]["config_note"]
    assert "states" in note and "custom_renderer" in note
    assert "pixels" in note


def test_ogbmaze_appearance_leaves_are_mapped_on_gate_evidence():
    """floor.color, floor.pixel_encoding and light.intensity are rendered and gated against swm.

    pixel_encoding keeps swm's sense and default, so it maps by identity."""
    m = _map_for(MAZE)
    for leaf in ("floor.color", "floor.pixel_encoding", "light.intensity"):
        assert m.pairs()[leaf] == (leaf, 1.0)
        rec = LEAF_CLASSIFICATION[MAZE]["leaves"][leaf]
        assert rec["classification"] == "mapped" and "appearance_gate" in rec["evidence"]
    assert "DEFAULTS DIFFER" not in m.notes["floor.pixel_encoding"]
    assert not m.unmapped()


@pytest.mark.parametrize("cell", sorted(RECORDED_OURS_LEAVES))
def test_every_own_side_leaf_is_mapped_or_carries_an_ours_only_reason(cell):
    """Every own-side leaf is mapped (a key of `pairs()`) or carries a non-empty `ours_only` reason.

    `ours_only` may not name a leaf our space does not have.
    """
    m = NEWCELL_VARIATION_MAPS[cell]
    ours = set(RECORDED_OURS_LEAVES[cell])
    mapped = set(m.pairs())
    declared = set(m.ours_only)
    unexplained = ours - mapped - declared
    assert not unexplained, (
        f"{cell}: {sorted(unexplained)} are leaves of our variation space and appear in neither the map nor ours_only"
    )
    both = mapped & declared
    assert not both, f"{cell}: {sorted(both)} are both mapped and declared ours-only"
    assert not (declared - ours), (
        f"{cell}: ours_only names {sorted(declared - ours)}, which our variation space does not offer"
    )
    for leaf in declared:
        assert m.ours_only[leaf].strip(), f"{cell}:{leaf} has an empty ours_only reason"


@pytest.mark.parametrize("cell", sorted(RECORDED_OURS_LEAVES))
def test_no_ours_only_leaf_is_secretly_offered_by_swm(cell):
    """An `ours_only` leaf swm ALSO offers is a mapping we failed to make, not a leaf they lack."""
    m = NEWCELL_VARIATION_MAPS[cell]
    theirs = set(RECORDED_SWM_LEAVES[m.swm_id])
    for leaf in m.ours_only:
        assert leaf not in theirs, f"{cell}:{leaf} is declared ours-only but swm offers it too"


def test_ogbmaze_ours_only_names_the_five_leaves_swm_has_no_counterpart_for():
    """The exact ours_only inventory for OGBMaze."""
    m = _map_for(MAZE)
    assert set(m.ours_only) == {
        "agent.scale",
        "floor.tint",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
    }
    assert "checker" in m.ours_only["floor.tint"], (
        "floor.tint's reason must say why it is not swm's floor.color (ours multiplies the bound "
        "texture, swm replaces the checker's two RGB values)"
    )
    assert "pixel_encoding" in m.ours_only["rendering.decolor_floor"], (
        "decolor_floor's reason must name pixel_encoding, which is the same switch"
    )


def test_both_pointmaze_large_cells_share_one_mapping():
    """Navigate and stitch differ by dataset, not appearance, so they share one mapping."""
    a = NEWCELL_VARIATION_MAPS["PointmazeLargeNavigate"]
    b = NEWCELL_VARIATION_MAPS["PointmazeLargeStitch"]
    assert a.swm_id == b.swm_id == MAZE
    assert a.pairs() == b.pairs() and dict(a.swm_only) == dict(b.swm_only)


def test_simple_point_maze_maps_appearance_speed_and_start():
    m = _map_for(SPM)
    assert set(m.pairs()) == {
        "agent.color",
        "agent.radius",
        "agent.speed",
        "agent.position",
        "background.color",
        "walls.color",
        "walls.number",
        "walls.shape",
        "walls.positions",
        "goal.color",
        "goal.radius",
        "goal.position",
    }
    for ours, (theirs, factor) in m.pairs().items():
        assert ours == theirs and factor == 1.0


def test_simple_point_maze_goal_appearance_is_mapped():
    for leaf in ("goal.color", "goal.radius"):
        assert LEAF_CLASSIFICATION[SPM]["leaves"][leaf]["classification"] == "mapped"
    assert "show_goal=True" in LEAF_CLASSIFICATION[SPM]["config_note"]


@pytest.mark.parametrize("leaf", ["agent.position", "goal.position"])
def test_simple_point_maze_positions_are_reset_leaves_not_style(leaf):
    spm = pytest.importorskip("stable_envs.envs.navigation.envs")
    assert LEAF_CLASSIFICATION[SPM]["leaves"][leaf]["classification"] == "mapped"
    assert leaf in {reset.path for reset in spm.SimplePointMaze.reset_leaves}
    assert leaf not in spm.SimplePointMaze.make_variation_space(spm.SimplePointMaze).names()


def test_simple_point_maze_mapped_leaves_are_dims_of_our_own_space():
    """Every mapped path is a leaf of our variation space or one of our reset leaves."""
    spm = pytest.importorskip("stable_envs.envs.navigation.envs")
    space = spm.SimplePointMaze.make_variation_space(spm.SimplePointMaze)
    ours = set(space.names()) | {leaf.path for leaf in spm.SimplePointMaze.reset_leaves}
    for leaf in _map_for(SPM).pairs():
        assert leaf in ours, f"{leaf} is mapped but is not a leaf of SimplePointMaze's own space"


# ---------------------------------------------------------------- OGB cube.color, on the CPU
#
# These run the real `_cube_colours`, `_cube_geom_index` and `_upload_cube_colours` against a stub
# bundle and a stub mujoco_warp context, so they need no GPU. The render context is reused across
# renders, so every render must upload the cube color, including the default.

_CUBE_DEFAULT = [0.96, 0.26, 0.33]
_CUBE_GEOMS = [62, 63]
#: object_0 is opaque; target_object_0 is the translucent goal marker. Only RGB is written.
_CUBE_ALPHAS = [1.0, 0.2]
#: swm's declared checker default; only its width matters to these tests.
_FLOOR_DEFAULT = [[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]


def _stub_env(device="cpu"):
    """An `OGBManipBase` with only the attributes the color helpers touch."""
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
    """A stand-in for the cached mujoco_warp context: one mutable per-world `geom_rgba`, reused across renders."""
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
    """One render's host-side style, produced by the real `_style_host`."""
    env = object.__new__(_base_cls())
    return env._style_host(_StubStyle(rows))


def _base_cls():
    from stable_envs.envs.ogbench.manipulation import OGBManipBase

    return OGBManipBase


def _upload(env, c, sty, B, nworld):
    """One render's worth of uploads, chunked exactly as `render` chunks them."""
    from stable_envs.envs.ogbench.manipulation import mjw_worlds

    chunk = mjw_worlds()
    cube = env._cube_colours(sty, B)
    assert cube is not None, "a cell with cube geoms must always be given a colour to write"
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        env._upload_cube_colours(c, cube[lo:hi], hi - lo)


def _rows(buf, n):
    return buf[:n][:, _CUBE_GEOMS, :].numpy()


def _expect(colours):
    out = np.zeros((len(colours), len(_CUBE_GEOMS), 4), np.float64)
    for i, rgb in enumerate(colours):
        out[i, :, :3] = rgb
        out[i, :, 3] = _CUBE_ALPHAS
    return out


def test_custom_then_default_restores_the_default_rgb():
    """A default render after a custom one must not keep the previous color, and returns the first default exactly."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    _upload(env, c, _sty([_CUBE_DEFAULT] * 4), 4, 4)
    first = _rows(buf, 4).copy()
    _upload(env, c, _sty([[1.0, 0.0, 0.0]] * 4), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect([[1.0, 0.0, 0.0]] * 4))
    _upload(env, c, _sty([_CUBE_DEFAULT] * 4), 4, 4)
    assert np.array_equal(_rows(buf, 4), first)
    assert np.allclose(first, _expect([_CUBE_DEFAULT] * 4))


def test_mixed_batch_then_default_restores_every_world():
    """After a per-environment style, a uniform default restores every world, not just world 0."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    mixed = [[1.0, 0.0, 0.0], _CUBE_DEFAULT, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    _upload(env, c, _sty(mixed), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect(mixed))
    _upload(env, c, _sty([_CUBE_DEFAULT] * 4), 4, 4)
    assert np.allclose(_rows(buf, 4), _expect([_CUBE_DEFAULT] * 4))


def test_a_small_default_batch_after_a_large_custom_one_leaves_no_stale_world():
    """A render of 2 worlds writes worlds 0-1 of a buffer an earlier render filled to 4; those must be correct."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(4, ngeom)
    _upload(env, c, _sty([[1.0, 0.0, 1.0]] * 4), 4, 4)
    _upload(env, c, _sty([_CUBE_DEFAULT] * 2), 2, 4)
    assert np.allclose(_rows(buf, 2), _expect([_CUBE_DEFAULT] * 2))


def test_each_cube_geom_keeps_its_own_alpha():
    """The opaque cube and the translucent goal marker are not flattened to one alpha by the color write."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(1, ngeom)
    _upload(env, c, _sty([[0.2, 0.4, 0.6]]), 1, 1)
    assert list(buf[0][_CUBE_GEOMS][:, 3].numpy()) == pytest.approx(_CUBE_ALPHAS)


def test_the_colour_write_touches_only_the_cube_geoms():
    """Every other geom row of the shared buffer is untouched."""
    env, ngeom = _stub_env()
    c, buf = _stub_context(2, ngeom)
    before = buf.clone()
    _upload(env, c, _sty([[1.0, 0.0, 0.0]] * 2), 2, 2)
    other = [g for g in range(ngeom) if g not in _CUBE_GEOMS]
    assert torch.equal(buf[:, other], before[:, other])


# ---------------------------------------------------------------- the cube.color shape contract


def _ogb_space(cell):
    """The OGB manipulation variation space for ``cell``, built without CUDA."""
    import types

    pytest.importorskip("ogbench")
    from stable_envs.envs.ogbench.manipulation import OGBManipBase
    from stable_envs.envs.ogbench.manipulation_renderer import configured_bundle

    token, _ = configured_bundle(cell)
    return OGBManipBase.make_variation_space(types.SimpleNamespace(CELL=token))


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_frozen_cube_colour_leaf_is_flat_three_and_floor_colour_is_two_by_three(cell):
    """swm declares `cube.color` as `(num_cubes, 3)`; both cells have one cube, so the leaf is carried flat."""
    space = _ogb_space(cell)
    layout = {p: (o, w) for p, o, w in space.flat_layout()}
    assert space["cube"]["color"].shape == (3,)
    assert space["floor"]["color"].shape == (2, 3)
    assert layout["cube.color"][1] == 3 and layout["floor.color"][1] == 6


@pytest.mark.parametrize("cell,swm_id", [("Cube", "swm/OGBCube-v0"), ("Scene", "swm/OGBScene-v0")])
def test_a_swm_shaped_cube_colour_survives_the_round_trip_through_our_style_row(cell, swm_id):
    """swm's `(1, 3)` cube color lands in the width-3 style slot and comes back as the same three numbers."""
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


def test_the_whole_style_row_crosses_to_the_host_in_one_read():
    """`_style_host` returns every leaf from one device-to-host copy."""
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
    """The host copy is cached per tensor and version: an in-place write invalidates it, another tensor misses."""
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
