"""The twelve Fetch VariationMaps: coverage, units, isolation and the default mapping.

Tier 1 checks the map table and needs no dependency. Tier 2 checks the table's leaf snapshot
against the live stable-worldmodel wrapper, so an upstream leaf added or renamed fails here.
Tier 3 builds envs (mujoco_warp and CUDA) and checks the mapping against them.

The default-equality tests compare stable-envs' own ``default_style()`` with
``style_from_variation(env, {})`` and the ``style -> swm config -> style`` round trip. They state
that an empty swm config selects our registered default exactly. They are not equality with
the wrapper's reset appearance, which unbinds the table, floor and object materials on the first
visual leaf and so differs from the official appearance.

``STABLE_ENVS_REQUIRE_LIVE_SWM=1`` turns every skip in tiers 2 and 3 into a failure, so a
certification run cannot pass without exercising the live wrapper and the GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.helpers import reference as WM
from tests.helpers.reference.variation_maps import fetch as FM

SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
SUFFIXES = ("", "Dense", "Dict")
CLASSES = [f"Fetch{s}{x}" for s in SCENES for x in SUFFIXES]

#: Leaves of the live ``FetchWrapper``. Reach carries no ``block.*`` because it has no object;
#: everything else is identical across the twelve ids.
SWM_LEAVES = {
    "Reach": (
        "agent.start_position",
        "background.color",
        "camera.angle_delta",
        "goal.start_position",
        "light.intensity",
        "object.color",
        "rendering.transparent_arm",
        "table.color",
    ),
    "object": (
        "agent.start_position",
        "background.color",
        "block.angle",
        "block.mass",
        "block.start_position",
        "camera.angle_delta",
        "goal.start_position",
        "light.intensity",
        "object.color",
        "rendering.transparent_arm",
        "table.color",
    ),
}


def leaves_for(scene: str) -> tuple[str, ...]:
    return SWM_LEAVES["Reach"] if scene == "Reach" else SWM_LEAVES["object"]


# ============================================================== tier 1: the table, no dependencies


def test_every_registered_fetch_id_has_a_map():
    """All twelve, reachable through the name a caller uses."""
    for c in CLASSES:
        m = WM.variation_map(c)
        assert m is not None, f"{c} has no VariationMap"
        scene = c[len("Fetch") :].replace("Dense", "").replace("Dict", "")
        suffix = c[len("Fetch") + len(scene) :]
        assert m.swm_id == f"swm/Fetch{scene}{suffix}-v3"


def test_the_three_suffixes_of_a_scene_share_one_map_body():
    """Dense changes a reward and Dict an observation container; only ``swm_id`` differs."""
    for scene in SCENES:
        bodies = []
        for x in SUFFIXES:
            m = WM.variation_map(f"Fetch{scene}{x}")
            bodies.append(
                (m.identical, dict(m.renamed), dict(m.scaled), dict(m.partial), dict(m.unmapped()), dict(m.ours_only))
            )
        assert bodies[0] == bodies[1] == bodies[2], f"the three Fetch{scene} ids disagree"


def test_every_swm_leaf_is_classified_exactly_once():
    """No swm leaf is silently unmapped, and no entry names a leaf swm does not offer."""
    for c in CLASSES:
        m = WM.variation_map(c)
        scene = c[len("Fetch") :].replace("Dense", "").replace("Dict", "")
        leaves = set(leaves_for(scene))
        mapped = {t for t, _ in m.pairs().values()}
        declared = set(m.unmapped())
        assert not (mapped & declared), f"{c}: a leaf is both mapped and declared unmapped"
        assert set(m.partial) <= mapped, f"{c}: partial names leaves no mapping field reaches"
        assert leaves - mapped - declared == set(), (
            f"{c}: unclassified swm leaves {sorted(leaves - mapped - declared)}"
        )
        assert mapped - leaves == set(), f"{c}: maps onto leaves swm does not offer"
        assert declared - leaves == set(), f"{c}: declares reasons for leaves swm does not offer"


def test_reach_lacks_exactly_the_object_leaves():
    reach, push = WM.variation_map("FetchReach"), WM.variation_map("FetchPush")
    assert reach.classification()["object.color"] == "dead_upstream"
    assert set(push.identical) - set(reach.identical) == {
        "object.color",
        "block.start_position",
        "block.angle",
        "block.mass",
    }
    assert set(reach.unmapped()) - set(push.unmapped()) == {"object.color"}


def test_the_light_factor_is_exact_and_covers_the_whole_swm_range():
    """swm's ``light.intensity`` is absolute and ours multiplies the model's 0.8."""
    scaled = WM.variation_map("FetchReach").scaled["light.diffuse"]
    assert scaled[0] == "light.intensity"
    assert scaled[1] * FM.LIGHT_DIFFUSE_0 == 1.0, "swm's default 0.8 must round-trip to our 1.0 exactly"
    assert 1.0 * scaled[1] <= 2.0, "swm's Box(0, 1) must fit our Box(0, 2)"


def test_only_the_camera_angle_leaf_is_dead_upstream():
    m = WM.variation_map("FetchPush")
    cls = m.classification()
    assert {p for p, k in cls.items() if k != "mapped"} == {"camera.angle_delta"}
    assert cls["camera.angle_delta"] == "dead_upstream"
    assert not (m.swm_only or m.unsupported or m.not_applicable or m.partial)
    for c in CLASSES:
        assert set(WM.variation_map(c).replacement_class().values()) <= {"NATIVE", "DEAD_UPSTREAM"}, c


def test_the_dead_camera_leaf_is_not_quietly_mapped_onto_our_working_camera():
    """Our camera moves and the reference's free camera does not; mapping them would diverge."""
    for c in CLASSES:
        m = WM.variation_map(c)
        assert "camera.angle_delta" not in {t for t, _ in m.pairs().values()}
        assert set(m.ours_only) >= {"camera.azimuth_delta", "camera.elevation_delta", "camera.distance_scale"}


def test_background_drives_the_floor_and_the_sky():
    """swm's ``background`` is the floor plus the skybox; the floor maps last so the inverse reads it."""
    for c in CLASSES:
        m = WM.variation_map(c)
        assert m.renamed == {"sky.color": "background.color", "floor.color": "background.color"}, c
        assert list(m.renamed)[-1] == "floor.color"
        assert m.classification()["background.color"] == "mapped", c


# ====================================================== the opt-in strictness switch for tiers 2/3

#: Under ``STABLE_ENVS_REQUIRE_LIVE_SWM=1`` a missing dependency in tier 2 or 3 is a failure.
REQUIRE_LIVE = os.environ.get("STABLE_ENVS_REQUIRE_LIVE_SWM", "").strip() not in ("", "0", "false")


def _unavailable(why: str):
    """Skip, or fail when this run declared that these tiers must run."""
    if REQUIRE_LIVE:
        pytest.fail(f"STABLE_ENVS_REQUIRE_LIVE_SWM=1 and this tier could not run: {why}")
    pytest.skip(why)


def _need(module: str):
    """``importorskip`` that honors ``REQUIRE_LIVE``."""
    try:
        return pytest.importorskip(module)
    except pytest.skip.Exception:
        _unavailable(f"{module} is not importable")


# ============================================================== tier 2: the live swm wrapper


def _live_space(swm_id):
    _need("stable_worldmodel")
    _need("gymnasium_robotics")
    try:
        return FM.swm_variation_space(swm_id)
    except Exception as exc:  # pragma: no cover
        _unavailable(f"{swm_id} is not constructible on this interpreter: {exc!r}")


def _live_leaves(swm_id):
    return sorted(_live_space(swm_id).names())


@pytest.mark.swm
@pytest.mark.parametrize("cls", CLASSES)
def test_the_frozen_leaf_snapshot_is_still_what_the_wrapper_offers(cls):
    """An upstream leaf added or renamed fails here rather than becoming an unclassified leaf."""
    m = WM.variation_map(cls)
    scene = cls[len("Fetch") :].replace("Dense", "").replace("Dict", "")
    assert _live_leaves(m.swm_id) == sorted(leaves_for(scene))


@pytest.mark.swm
def test_check_mapping_reads_the_variation_space_through_the_fetch_wrapper():
    """``check_mapping`` must reach the swm space past ``FetchWrapper`` and classify every leaf."""
    _need("stable_worldmodel")
    _need("gymnasium_robotics")
    m = WM.variation_map("FetchPush")
    space = _live_space(m.swm_id)

    # A stand-in for our side (class name and ``variation_space.names()``) keeps this off the GPU.
    ours = tuple(sorted(set(m.pairs()) | set(m.ours_only)))
    env = type("FetchPush", (), {})()
    env.variation_space = type("_Space", (), {"names": staticmethod(lambda: ours)})()

    got = WM.check_mapping(env)
    assert "error" not in got, got
    assert got["swm_id"] == "swm/FetchPush-v3"
    assert got["undocumented"] == [], f"undocumented Fetch leaves: {got['undocumented']}"
    assert set(got["classification"]) == set(space.names())
    assert got["classification"]["camera.angle_delta"] == "dead_upstream"
    for leaf in ("background.color", "table.color", "rendering.transparent_arm", "block.mass"):
        assert got["classification"][leaf] == "mapped", leaf
    assert set(got["covered"]) == set(space.names()) - {"camera.angle_delta"}


# ============================================================== tier 3: against a built env


@pytest.fixture(scope="module")
def envs():
    """One built env per scene; the three suffixes share a renderer and a model."""
    torch = _need("torch")
    _need("mujoco_warp")
    if not torch.cuda.is_available():
        _unavailable("the Fetch family renders only through mujoco_warp on cuda")
    import stable_envs as se

    if "fetch" in se.import_errors():
        _unavailable(f"stable_envs.envs.fetch: {se.import_errors()['fetch']}")
    # The module imports gymnasium_robotics lazily, so a missing install surfaces at the first build.
    try:
        return {s: se.make(f"Fetch{s}", num_envs=2, size=64, device="cuda") for s in SCENES}
    except Exception as exc:
        _unavailable(f"this interpreter cannot build a Fetch env: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("scene", SCENES)
def test_the_mapped_and_ours_only_paths_are_real_leaves_of_our_space(envs, scene):
    """A map entry naming a path our space lacks would be silently ignored by ``pairs()``."""
    env = envs[scene]
    ours = set(env.variation_space.names())
    m = WM.variation_map(f"Fetch{scene}")
    resets = set(FM.STATE_LEAVES) | (set(FM.BLOCK_STATE_LEAVES) if scene != "Reach" else set())
    assert set(m.pairs()) - resets <= ours, f"{scene}: maps from paths we do not have: {set(m.pairs()) - ours}"
    assert set(m.ours_only) <= ours, f"{scene}: {set(m.ours_only) - ours}"
    assert ours - set(m.pairs()) - set(m.ours_only) == set(), (
        f"{scene}: our leaves with no entry in either direction: {sorted(ours - set(m.pairs()) - set(m.ours_only))}"
    )


@pytest.mark.parametrize("scene", SCENES)
def test_registered_default_is_byte_identical_through_the_mapping(envs, scene):
    """An empty swm config leaves our style exactly at the registered default (no tolerance)."""
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    got = WM.style_from_variation(env, {})
    assert got.dtype == default.dtype
    assert np.array_equal(got, default), f"{scene}: an empty variation moved the default style"


@pytest.mark.parametrize("scene", SCENES)
def test_the_default_style_round_trips_to_swm_units_and_back(envs, scene):
    """``style -> swm config -> style`` is the identity, except that the sky returns as the floor color."""
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    cfg = WM.variation_from_style(env, default)
    assert np.isclose(cfg["light.intensity"], FM.LIGHT_DIFFUSE_0), (
        "our default light multiplier must express swm's own 0.8, not its declared 0.7"
    )
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    o, w = layout["sky.color"]
    back = WM.style_from_variation(env, cfg)
    floor = layout["floor.color"]
    assert np.array_equal(back[o : o + w], default[floor[0] : floor[0] + floor[1]])
    back[o : o + w] = default[o : o + w]
    assert np.array_equal(back, default)


@pytest.mark.parametrize("scene", SCENES)
def test_a_mapped_leaf_changes_only_its_own_style_slot(envs, scene):
    """One swm leaf touches only the style slots mapped from it."""
    env = envs[scene]
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    default = np.asarray(env.default_style(), np.float32)
    probes = {
        "table.color": [0.9, 0.2, 0.05],
        "background.color": [0.9, 0.05, 0.9],
        "object.color": [0.05, 0.95, 0.2],
        "light.intensity": [0.55],
        "rendering.transparent_arm": 1,
        "block.mass": [40.0],
    }
    m = WM.variation_map(f"Fetch{scene}")
    for theirs, value in probes.items():
        mine = [ours for ours, (t, _) in m.pairs().items() if t == theirs and ours in layout]
        if not mine:
            continue
        row = WM.style_from_variation(env, {theirs: value})
        rest = np.ones(row.shape, bool)
        for ours in mine:
            o, w = layout[ours]
            assert not np.array_equal(row[o : o + w], default[o : o + w]), f"{theirs} moved nothing in {ours}"
            rest[o : o + w] = False
        assert np.array_equal(row[rest], default[rest]), f"{scene}: {theirs} changed style slots other than {mine}"


@pytest.mark.parametrize("scene", SCENES)
def test_an_unmapped_leaf_moves_nothing(envs, scene):
    """A dead leaf or a State reset option does not leak into the style."""
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    values = {
        "camera.angle_delta": [[10.0, 10.0]],
        "agent.start_position": [1.45, 0.9],
        "goal.start_position": [1.45, 0.9, 0.4247],
        "block.start_position": [1.45, 0.9],
        "block.angle": [1.5],
    }
    m = WM.variation_map(f"Fetch{scene}")
    for leaf in set(m.unmapped()) | set(FM.STATE_LEAVES) | set(FM.BLOCK_STATE_LEAVES):
        if leaf not in values or (scene == "Reach" and leaf.startswith("block.")):
            continue
        assert np.array_equal(WM.style_from_variation(env, {leaf: values[leaf]}), default), (
            f"{scene}: {leaf} moved our style"
        )


@pytest.mark.parametrize("scene", SCENES)
def test_strict_mode_accepts_a_full_swm_config(envs, scene):
    """A complete swm config passes ``strict=True``, which raises on a leaf with no entry."""
    env = envs[scene]
    cfg = {
        "table.color": [0.3, 0.3, 0.3],
        "object.color": [0.8, 0.1, 0.1],
        "background.color": [0.1, 0.1, 0.1],
        "light.intensity": [0.7],
        "camera.angle_delta": [[0.0, 0.0]],
        "rendering.transparent_arm": 0,
        "agent.start_position": [1.3418, 0.7491],
        "goal.start_position": [1.3, 0.74, 0.4247],
        **(
            {}
            if scene == "Reach"
            else {"block.start_position": [1.3, 0.74], "block.angle": [0.0], "block.mass": [2.0]}
        ),
    }
    WM.style_from_variation(env, cfg, strict=True)  # must not raise
