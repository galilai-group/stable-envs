"""The twelve Fetch VariationMaps: coverage, units, isolation, and the default that must not move.

Three tiers, and the split is deliberate rather than defensive:

**Tier 1 — the map is data, so most of it can be checked with nothing installed.** That the twelve
ids exist, that the three suffixes of a scene share one map body, that every swm leaf is classified
exactly once, and that the light factor is the exact binary 1.25 are properties of the table. They
run on any interpreter, which is the only reason they run at all in CI: this family needs
``mujoco_warp`` and a CUDA device to make a pixel.

**Tier 2 — the frozen leaf snapshot, re-enumerated from the live wrapper when it is present.** The
snapshot below was read off ``stable_worldmodel`` 176d545 and is what tier 1 checks against; tier 2
asserts that the snapshot still IS the live wrapper's answer, so an upstream leaf added or renamed
fails here loudly instead of silently becoming an unclassified leaf.

**Tier 3 — the mapping against a built env.** Needs ``stable_envs.envs.fetch``, i.e. ``mujoco_warp``
and CUDA; skipped everywhere else. This is where "the registered default is byte-identical through
the mapping" and "a mapped leaf changes only its own semantics" are actually asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from stable_envs.compat import worldmodel as WM
from stable_envs.compat.variation_maps import fetch as FM

SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
SUFFIXES = ("", "Dense", "Dict")
CLASSES = [f"Fetch{s}{x}" for s in SCENES for x in SUFFIXES]

#: Enumerated from the live ``FetchWrapper`` (stable_worldmodel 176d545, gymnasium-robotics 1.4.2,
#: mujoco 3.12.0) — see ``examples/results/step8/variations-fetch/leaf_classification.json``. Reach
#: carries no ``block.*`` because it has no object; everything else is identical across the twelve.
SWM_LEAVES = {
    "Reach": ("agent.start_position", "background.color", "camera.angle_delta",
              "goal.start_position", "light.intensity", "object.color",
              "rendering.transparent_arm", "table.color"),
    "object": ("agent.start_position", "background.color", "block.angle", "block.mass",
               "block.start_position", "camera.angle_delta", "goal.start_position",
               "light.intensity", "object.color", "rendering.transparent_arm", "table.color"),
}


def leaves_for(scene: str) -> tuple[str, ...]:
    return SWM_LEAVES["Reach"] if scene == "Reach" else SWM_LEAVES["object"]


# ============================================================== tier 1: the table, no dependencies


def test_every_registered_fetch_id_has_a_map():
    """All twelve, reachable through the name a caller actually uses.

    ``VARIATION_MAPS`` rather than ``FM.MAPS``, because the one-line registration in
    ``compat.worldmodel`` is half the deliverable and a module nobody imports registers nothing.
    """
    for c in CLASSES:
        m = WM.variation_map(c)
        assert m is not None, f"{c} has no VariationMap"
        scene = c[len("Fetch"):].replace("Dense", "").replace("Dict", "")
        suffix = c[len("Fetch") + len(scene):]
        assert m.swm_id == f"swm/Fetch{scene}{suffix}-v3"


def test_the_three_suffixes_of_a_scene_share_one_map_body():
    """Dense changes a reward and Dict an observation container; neither reaches a pixel.

    Twelve hand-written tables would be twelve places for the family to drift. This asserts the
    thing the module claims: the only field that differs between ``FetchPush``, ``FetchPushDense``
    and ``FetchPushDict`` is ``swm_id``.
    """
    for scene in SCENES:
        bodies = []
        for x in SUFFIXES:
            m = WM.variation_map(f"Fetch{scene}{x}")
            bodies.append((m.identical, dict(m.renamed), dict(m.scaled), dict(m.swm_only),
                           dict(m.ours_only)))
        assert bodies[0] == bodies[1] == bodies[2], f"the three Fetch{scene} ids disagree"


def test_every_swm_leaf_is_classified_exactly_once():
    """The property the whole exercise exists to establish: no leaf is silently unmapped."""
    for c in CLASSES:
        m = WM.variation_map(c)
        scene = c[len("Fetch"):].replace("Dense", "").replace("Dict", "")
        leaves = set(leaves_for(scene))
        mapped = {t for t, _ in m.pairs().values()}
        declared = set(m.swm_only)
        assert not (mapped & declared), f"{c}: a leaf is both mapped and declared unmapped"
        assert leaves - mapped - declared == set(), (
            f"{c}: unclassified swm leaves {sorted(leaves - mapped - declared)}")
        assert mapped - leaves == set(), f"{c}: maps onto leaves swm does not offer"
        assert declared - leaves == set(), f"{c}: declares reasons for leaves swm does not offer"


def test_reach_differs_from_the_object_scenes_in_exactly_two_ways():
    """The per-task difference is declarative, so it should be small and nameable."""
    reach, push = WM.variation_map("FetchReach"), WM.variation_map("FetchPush")
    assert "object.color" in push.identical and "object.color" not in reach.identical
    assert "object.color" in reach.swm_only and "DEAD UPSTREAM" in reach.swm_only["object.color"]
    assert set(push.swm_only) - set(reach.swm_only) == {"block.start_position", "block.angle",
                                                        "block.mass"}
    assert set(reach.swm_only) - set(push.swm_only) == {"object.color"}


def test_the_light_factor_is_exact_and_covers_the_whole_swm_range():
    """swm's ``light.intensity`` is ABSOLUTE and ours is a multiplier on the model's 0.8.

    Two separate claims. The factor must be exactly representable, or swm's default 0.8 would not
    round-trip to our default 1.0 and ``test_registered_default_is_byte_identical`` would fail by a
    ULP rather than by a mistake. And the full swm range must fit ours, because "we support this
    leaf" is false if half of it clips.
    """
    ours, factor = "light.diffuse", WM.variation_map("FetchReach").scaled["light.diffuse"][1]
    assert WM.variation_map("FetchReach").scaled[ours][0] == "light.intensity"
    assert factor == 1.25 and factor * FM.LIGHT_DIFFUSE_0 == 1.0     # exact, not approximate
    assert 0.0 * factor >= 0.0 and 1.0 * factor <= 2.0               # swm Box(0,1) -> our Box(0,2)


def test_the_dead_and_unsupported_leaves_carry_their_mechanism():
    """A reason that does not name a mechanism is a shrug with a docstring."""
    m = WM.variation_map("FetchPush")
    assert "DEAD UPSTREAM" in m.swm_only["camera.angle_delta"]
    assert "FREE camera" in m.swm_only["camera.angle_delta"]
    assert "UNSUPPORTED BY BACKEND" in m.swm_only["rendering.transparent_arm"]
    assert "alpha" in m.swm_only["rendering.transparent_arm"]
    assert "NOT APPLICABLE" in m.swm_only["block.mass"]
    for leaf in ("agent.start_position", "goal.start_position", "block.start_position",
                 "block.angle"):
        assert "STATE" in m.swm_only[leaf], leaf


def test_the_dead_camera_leaf_is_not_quietly_mapped_onto_our_working_camera():
    """We have a camera that moves; the reference's does not. Mapping them would be a divergence."""
    for c in CLASSES:
        m = WM.variation_map(c)
        assert "camera.angle_delta" not in {t for t, _ in m.pairs().values()}
        assert set(m.ours_only) >= {"camera.azimuth_delta", "camera.elevation_delta",
                                    "camera.distance_scale"}


def test_background_maps_to_the_floor_and_says_what_it_leaves_out():
    """swm's "background" is floor geoms PLUS a skybox texture; we render only the floor half."""
    m = WM.variation_map("FetchSlide")
    assert m.renamed["floor.color"] == "background.color"
    assert "skybox" in FM.__doc__ and "render context per value" in FM.__doc__


# ============================================================== tier 2: the live swm wrapper


def _live_leaves(swm_id):
    pytest.importorskip("stable_worldmodel")
    pytest.importorskip("gymnasium_robotics")
    try:
        return sorted(FM.swm_variation_space(swm_id).names())
    except Exception as exc:                                          # pragma: no cover
        pytest.skip(f"{swm_id} is not constructible on this interpreter: {exc!r}")


@pytest.mark.parametrize("cls", CLASSES)
def test_the_frozen_leaf_snapshot_is_still_what_the_wrapper_offers(cls):
    """An upstream leaf added or renamed must fail HERE, not become an unclassified leaf."""
    m = WM.variation_map(cls)
    scene = cls[len("Fetch"):].replace("Dense", "").replace("Dict", "")
    assert _live_leaves(m.swm_id) == sorted(leaves_for(scene))


@pytest.mark.parametrize("cls", CLASSES)
def test_no_live_leaf_is_undocumented(cls):
    m = WM.variation_map(cls)
    live = set(_live_leaves(m.swm_id))
    assert live - {t for t, _ in m.pairs().values()} - set(m.swm_only) == set()


# ============================================================== tier 3: against a built env


@pytest.fixture(scope="module")
def envs():
    """One built env per SCENE (not per id): the three suffixes share a renderer and a model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("the Fetch family renders only through mujoco_warp on cuda")
    import stable_envs as se
    if "fetch" in se.envs.import_errors():
        pytest.skip(f"stable_envs.envs.fetch: {se.envs.import_errors()['fetch']}")
    return {s: se.make(f"Fetch{s}", batch=2, size=64, device="cuda") for s in SCENES}


@pytest.mark.parametrize("scene", SCENES)
def test_the_mapped_and_ours_only_paths_are_real_leaves_of_our_space(envs, scene):
    """A map entry naming a path our space does not have is silently ignored by ``pairs()``."""
    env = envs[scene]
    ours = set(env.variation_space.names())
    m = WM.variation_map(f"Fetch{scene}")
    assert set(m.pairs()) <= ours, f"{scene}: maps from paths we do not have: {set(m.pairs()) - ours}"
    assert set(m.ours_only) <= ours, f"{scene}: {set(m.ours_only) - ours}"
    assert ours - set(m.pairs()) - set(m.ours_only) == set(), (
        f"{scene}: our leaves with no entry in either direction: "
        f"{sorted(ours - set(m.pairs()) - set(m.ours_only))}")


@pytest.mark.parametrize("scene", SCENES)
def test_registered_default_is_byte_identical_through_the_mapping(envs, scene):
    """An empty swm config must leave our style EXACTLY at the official appearance.

    ``allclose`` would hide the interesting failure. The reason this can be byte-exact at all is that
    every mapped leaf either has factor 1.0 or has the exactly representable 1.25.
    """
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    got = WM.style_from_variation(env, {})
    assert got.dtype == default.dtype
    assert np.array_equal(got, default), f"{scene}: an empty variation moved the default style"


@pytest.mark.parametrize("scene", SCENES)
def test_the_default_style_round_trips_to_swm_units_and_back(envs, scene):
    """``style -> swm config -> style`` is the identity, including the 1.25 light factor."""
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    cfg = WM.variation_from_style(env, default)
    assert np.isclose(cfg["light.intensity"], FM.LIGHT_DIFFUSE_0), (
        "our default light multiplier must express swm's own 0.8, not its declared 0.7")
    assert np.array_equal(WM.style_from_variation(env, cfg), default)


@pytest.mark.parametrize("scene", SCENES)
def test_a_mapped_leaf_changes_only_its_own_style_slot(envs, scene):
    """Isolation, at the level the mapping owns: one swm leaf touches one of our leaves."""
    env = envs[scene]
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    default = np.asarray(env.default_style(), np.float32)
    probes = {"table.color": [0.9, 0.2, 0.05], "background.color": [0.9, 0.05, 0.9],
              "object.color": [0.05, 0.95, 0.2], "light.intensity": [0.55]}
    m = WM.variation_map(f"Fetch{scene}")
    inv = {theirs: ours for ours, (theirs, _) in m.pairs().items()}
    for theirs, value in probes.items():
        if theirs not in inv:
            continue
        ours = inv[theirs]
        row = WM.style_from_variation(env, {theirs: value})
        o, w = layout[ours]
        assert not np.array_equal(row[o:o + w], default[o:o + w]), f"{theirs} moved nothing"
        rest = np.ones(row.shape, bool)
        rest[o:o + w] = False
        assert np.array_equal(row[rest], default[rest]), (
            f"{scene}: {theirs} changed style slots other than {ours}")


@pytest.mark.parametrize("scene", SCENES)
def test_an_unmapped_leaf_moves_nothing(envs, scene):
    """The other half of isolation: a leaf we declare unsupported must not leak into the style."""
    env = envs[scene]
    default = np.asarray(env.default_style(), np.float32)
    values = {"camera.angle_delta": [[10.0, 10.0]], "rendering.transparent_arm": 1,
              "block.mass": [40.0], "agent.start_position": [1.45, 0.9],
              "goal.start_position": [1.45, 0.9, 0.4247], "block.start_position": [1.45, 0.9],
              "block.angle": [1.5]}
    m = WM.variation_map(f"Fetch{scene}")
    for leaf in m.swm_only:
        if leaf not in values:                       # object.color on Reach: dead, and not a style
            continue
        assert np.array_equal(WM.style_from_variation(env, {leaf: values[leaf]}), default), (
            f"{scene}: the unmapped leaf {leaf} moved our style")


@pytest.mark.parametrize("scene", SCENES)
def test_strict_mode_accepts_a_full_swm_config(envs, scene):
    """``strict=True`` raises on a leaf with no entry. A complete swm config must pass it."""
    env = envs[scene]
    cfg = {"table.color": [0.3, 0.3, 0.3], "object.color": [0.8, 0.1, 0.1],
           "background.color": [0.1, 0.1, 0.1], "light.intensity": [0.7],
           "camera.angle_delta": [[0.0, 0.0]], "rendering.transparent_arm": 0,
           "agent.start_position": [1.3418, 0.7491], "goal.start_position": [1.3, 0.74, 0.4247],
           **({} if scene == "Reach" else {"block.start_position": [1.3, 0.74],
                                           "block.angle": [0.0], "block.mass": [2.0]})}
    WM.style_from_variation(env, cfg, strict=True)        # must not raise
