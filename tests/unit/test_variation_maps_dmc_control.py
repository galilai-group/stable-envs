"""The variation mapping for the five classic-control and thirteen DM Control cells.

What each test is FOR, because a variation map fails in ways a smoke test cannot see:

``test_every_leaf_is_classified``
    the completeness claim. Read against the LIVE swm variation space, not against a transcribed
    list, so a leaf swm adds tomorrow fails here instead of being silently unmapped.
``test_registered_default_*``
    Certification asks that "the registered default is reproducible through the mapping". For most
    cells the mapped default IS the cell's default style and the render is byte-identical; for the
    cells in ``DEFAULT_REACHED_THROUGH`` the swm default is a DIFFERENT appearance from the compiled
    model's and is reached through named leaves. Both are asserted, and the second names the leaves,
    so a cell that quietly starts needing another leaf fails.
``test_*_isolation``
    a mapped leaf must move its own style slots and nothing else.
``test_round_trip``
    swm value -> style row -> swm value.
``test_rgb255_is_swm_quantisation``
    the one arithmetic claim in the classic-control mapping.
``test_light_intensity_lands_on_the_model``
    the DM Control mapping's only non-identity conversion, checked where it matters — on the compiled
    model's ``light_diffuse`` — rather than on the style row.
``test_dead_upstream_leaves_really_are_dead``
    every ``dead_upstream`` classification re-measured against live swm, because that classification
    is a claim about somebody else's code and is the one most likely to rot.

The swm-dependent tests skip when ``stable_worldmodel`` is not importable, which is the case in the
repository's default test venv; they are run in a venv that has it (see
``pre-repository-cleanup:docs/research/step8_variations_dmc_control.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.registry import REGISTRY, make
from tests.helpers.reference import (
    VARIATION_MAPS,
    check_mapping,
    style_from_variation,
    swm_variation_space,
    variation_from_style,
    variation_map,
)
from tests.helpers.reference.variation_maps.dmc_control import (
    EXPECTED_COUNTS,
    EXPECTED_COUNTS_BY_CELL,
    LIGHT_RANGE,
    MAPS,
    swm_defaults,
)

_HAVE_SWM = True
try:  # pragma: no cover - environment dependent
    import gymnasium  # noqa: F401
    import stable_worldmodel  # noqa: F401
except Exception:  # pragma: no cover
    _HAVE_SWM = False

needs_swm = pytest.mark.skipif(not _HAVE_SWM, reason="stable_worldmodel is not installed")
#: The DM Control cells register only when dm_control and a GL backend are importable; the two
#: tests below are the only ones that need a DM Control cell and no swm.
needs_dmc = pytest.mark.skipif(
    "DMCManipulatorBringBall" not in REGISTRY, reason="the DM Control cells did not register (dm_control/MUJOCO_GL)"
)

CELLS = sorted(MAPS)
#: The cells whose swm REGISTERED default is a different appearance from the compiled model's,
#: and the exact style leaves the mapping moves to reach it. Measured; see the research doc.
DEFAULT_REACHED_THROUGH = {
    "DMCBallInCupCatch": {"light.diffuse_scale"},
    "DMCFingerTurnHard": {"light.diffuse_scale"},
    "DMCManipulatorBringBall": {"light.diffuse_scale", "target_ball.shape", "visibility.target_ball"},
}


def _env(cell):
    return make(cell, num_envs=1, device="cpu")


def _layout(env):
    return {p: (o, w) for p, o, w in env.variation_space.flat_layout()}


# --------------------------------------------------------------------- completeness


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_every_leaf_is_classified(cell):
    """No swm leaf is silently unmapped, and none of our leaves is undocumented either."""
    env = _env(cell)
    r = check_mapping(env)
    assert r["undocumented"] == [], f"{cell}: unclassified swm leaves {r['undocumented']}"
    assert r["mapped_but_absent_from_swm"] == [], (
        f"{cell}: map names swm leaves that do not exist: {r['mapped_but_absent_from_swm']}"
    )
    assert set(r["ours_only"]) == set(r["ours_only_documented"]), (
        f"{cell}: ours_only is out of step with its documentation"
    )
    assert all(v for v in r["reasons"].values()), f"{cell}: an unmapped leaf has an empty reason"


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_classification_counts_match_the_declaration(cell):
    """The four-way counts this module claims, re-derived from the live swm space."""
    env = _env(cell)
    r = check_mapping(env)
    got = {"total": len(r["classification"])}
    for label in ("mapped", "unsupported", "dead_upstream", "not_applicable"):
        got[label] = sum(1 for v in r["classification"].values() if v == label)
    assert got == EXPECTED_COUNTS_BY_CELL.get(cell, EXPECTED_COUNTS[variation_map(env).swm_id]), cell


@needs_swm
def test_the_seventeen_ids_total_161_leaves():
    """The headline count, so a change to any one cell is visible in one place."""
    seen, totals = set(), {"mapped": 0, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0}
    for cell in CELLS:
        m = variation_map(cell)
        if m.swm_id in seen:
            continue
        seen.add(m.swm_id)
        for k in totals:
            totals[k] += EXPECTED_COUNTS[m.swm_id][k]
    assert len(seen) == 17
    assert totals == {"mapped": 140, "unsupported": 0, "dead_upstream": 21, "not_applicable": 0}
    assert sum(totals.values()) == 161


# --------------------------------------------------------------------- the registered default


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_registered_default_is_reproducible_through_the_mapping(cell):
    """``style_from_variation`` at swm's own init_values, rendered against the cell's default style.

    Byte-identical where the two appearances coincide; where they do not, the difference is confined
    to the leaves named in ``DEFAULT_REACHED_THROUGH`` and nothing else moves.
    """
    env = _env(cell)
    base = np.asarray(env.default_style(), np.float32).reshape(-1)
    mapped = style_from_variation(env, swm_defaults(variation_map(env).swm_id))
    moved = {p for p, (o, w) in _layout(env).items() if not np.array_equal(base[o : o + w], mapped[o : o + w])}
    assert moved == DEFAULT_REACHED_THROUGH.get(cell, set()), cell

    states = torch.as_tensor(env.sample_states(8, rng=np.random.default_rng(3)))
    a = env._render(states, style=torch.as_tensor(np.repeat(base[None], 8, 0))).cpu().numpy()
    b = env._render(states, style=torch.as_tensor(np.repeat(mapped[None], 8, 0))).cpu().numpy()
    if cell not in DEFAULT_REACHED_THROUGH:
        assert np.array_equal(a, b), f"{cell}: the mapped swm default is not the cell's default frame"
    else:
        assert not np.array_equal(a, b), (
            f"{cell}: the swm default is declared to differ from the compiled model's and does not"
        )


@needs_swm
@pytest.mark.parametrize("cell", ["Acrobot", "CartPole", "Pendulum", "MountainCar", "MountainCarContinuous"])
def test_classic_default_matches_the_official_swm_render(cell):
    """The five classic-control cells, at the registered default, against swm's OWN frame.

    This is the one arm where the two libraries can be compared pixel for pixel: swm renders the
    official gymnasium env and then recolours it, so at the default (where every swap is a no-op) its
    frame is the official frame, and ours is certified against that. 0 disagreeing pixels over 8
    states is what makes "the registered default is reproducible" a measurement rather than a claim.
    """
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("pygame")
    env = _env(cell)
    cls = type(env)
    m = variation_map(env)
    states = env.sample_states(8, rng=np.random.default_rng(7))
    ref_wrapper = gym.make(m.swm_id, render_mode="rgb_array")
    try:
        holder = ref_wrapper
        while getattr(holder, "variation_space", None) is None:
            holder = holder.env
        holder.reset(seed=0)
        official = np.stack(
            [(cls.reference_set_state(holder.env.unwrapped, s), holder.render().copy())[1] for s in states]
        )
    finally:
        ref_wrapper.close()
    mapped = style_from_variation(env, swm_defaults(m.swm_id))
    ours = env._render(torch.as_tensor(states), style=torch.as_tensor(np.repeat(mapped[None], 8, 0))).cpu().numpy()
    assert np.array_equal(ours, official), f"{cell}: {int((ours != official).sum())} bytes differ"


# --------------------------------------------------------------------- isolation


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_mapped_leaves_are_isolated(cell):
    """Moving one swm leaf moves that leaf's style slots and no others.

    Asserted on the STYLE ROW rather than on pixels because that is where a mapping can be wrong: a
    pixel test cannot distinguish "the map wrote the wrong slot" from "two elements share a colour",
    and a slot test can. The one-to-many leaves (``visuals.fg``, ``visuals.cart``) are expected to
    move several slots and the expectation is derived from the map itself, not hard-coded.
    """
    env = _env(cell)
    m = variation_map(env)
    defaults = swm_defaults(m.swm_id)
    base = style_from_variation(env, defaults)
    layout = _layout(env)
    space = swm_variation_space(__import__("gymnasium").make(m.swm_id))
    for ours_paths, swm_path in _mapped_pairs(m):
        leaf = space
        for part in swm_path.split("."):
            leaf = leaf[part]
        probe = _probe(leaf)
        row = style_from_variation(env, {**defaults, swm_path: probe})
        moved = {p for p, (o, w) in layout.items() if not np.array_equal(base[o : o + w], row[o : o + w])}
        assert moved <= set(ours_paths), f"{cell}: swm {swm_path} moved {sorted(moved - set(ours_paths))} as well"


def _mapped_pairs(m):
    """``(our_paths, swm_path)`` for every mapped leaf, one row per swm leaf."""
    out: dict[str, list[str]] = {}
    for ours, (theirs, _) in m.pairs().items():
        out.setdefault(theirs, []).append(ours)
    for ours, (theirs, _, _) in m.affine.items():
        out.setdefault(theirs, []).append(ours)
    for ours, theirs in m.rgb255.items():
        out.setdefault(theirs, []).append(ours)
    for ours, (theirs, _) in m.enum.items():
        out.setdefault(theirs, []).append(ours)
    return [(v, k) for k, v in out.items()]


def _probe(leaf):
    """A value strictly inside the leaf's range and as far from its default as the range allows."""
    iv = np.asarray(leaf.init_value)
    if hasattr(leaf, "n"):
        return int((int(iv) + 1) % int(leaf.n))
    lo = np.asarray(leaf.low, np.float64).reshape(np.shape(iv))
    hi = np.asarray(leaf.high, np.float64).reshape(np.shape(iv))
    far = np.where(np.abs(iv - lo) > np.abs(iv - hi), lo, hi)
    return np.asarray(iv + 0.9 * (far - iv), leaf.dtype)


# --------------------------------------------------------------------- round trip


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_round_trip(cell):
    """swm value -> style row -> swm value, at the default and at a probe on every mapped leaf."""
    env = _env(cell)
    m = variation_map(env)
    defaults = swm_defaults(m.swm_id)
    space = swm_variation_space(__import__("gymnasium").make(m.swm_id))
    cfgs = [defaults]
    for _, swm_path in _mapped_pairs(m):
        leaf = space
        for part in swm_path.split("."):
            leaf = leaf[part]
        cfgs.append({**defaults, swm_path: _probe(leaf)})
    quantised = set(m.rgb255.values())
    for cfg in cfgs:
        back = variation_from_style(env, style_from_variation(env, cfg))
        for path in m.mapped_swm_paths():
            want = np.asarray(cfg[path], np.float64).reshape(-1)
            got = np.asarray(back[path], np.float64).reshape(-1)
            if path in quantised:
                # A colour we hold as 8 bits cannot return a float unchanged, and the loss is exactly
                # the quantisation: half a level, 1/510. The stronger statement is that it round-trips
                # to the SAME uint8, i.e. swm would paint the identical pixel, so that is asserted too.
                assert np.allclose(got, want, atol=1.0 / 510.0), f"{cell}: {path} {got} != {want}"
                assert np.array_equal(
                    (got * 255.0 + 0.5).astype(np.uint8), (np.clip(want, 0, 1) * 255.0 + 0.5).astype(np.uint8)
                ), f"{cell}: {path} does not round-trip to the same uint8"
            else:
                # float32 is the style row's dtype, so this is the representation and not slack.
                assert np.allclose(got, want, rtol=6e-8, atol=3e-7), f"{cell}: {path} {got} != {want}"


def test_rgb255_is_swm_quantisation():
    """``rgb255`` must reproduce ``GymControlWrapper.render``'s ``uint8(v * 255 + 0.5)`` exactly.

    A bare factor of 255 does not: ``to_u8`` truncates, so swm's 0.3 -> 77 becomes 76. Checked over a
    dense grid rather than at the endpoints, because the disagreement is at the half-integers.
    """
    env = make("Pendulum", num_envs=1, device="cpu")
    layout = _layout(env)
    o, w = layout["rod.color"]
    for t in np.linspace(0.0, 1.0, 513):
        row = style_from_variation(env, {"visuals.rod": np.full(3, t, np.float32)})
        want = (np.float32(t) * 255.0 + 0.5).astype(np.uint8)
        got = np.asarray(row[o : o + w]).astype(np.uint8)
        assert (got == want).all(), f"t={t}: {got} != {want}"


@needs_swm
@pytest.mark.parametrize("cell", sorted(LIGHT_RANGE))
def test_light_intensity_lands_on_the_model(cell):
    """The only non-identity DM Control conversion, checked on ``light_diffuse`` itself.

    swm writes the intensity ABSOLUTELY; ours multiplies the compiled model's own diffuse and clips.
    The test asserts the two agree over the reachable range, and that the declared low-end gap is
    real rather than a rounding artefact.
    """
    env = _env(cell)
    base, lo, hi, _ = LIGHT_RANGE[cell]
    pristine = np.asarray(type(env)._pristine()["light_diffuse"], np.float64)
    assert np.allclose(pristine, base), f"{cell}: declared light base {base} is not the model's"
    o, w = _layout(env)["light.diffuse_scale"]
    for intensity in np.linspace(lo, hi, 25):
        row = style_from_variation(env, {"light.intensity": np.float32([intensity])})
        got = np.clip(pristine * float(row[o]), 0.0, 1.0)
        assert np.allclose(got, intensity, atol=3e-7), f"{cell}: {got} != {intensity}"
    # and the gap: swm's darkest tenth of the range cannot be reached, because our scale stops at 0.2
    assert lo > 0.0
    dark = style_from_variation(env, {"light.intensity": np.float32([lo / 2.0])})
    assert (
        float(dark[o]) < float(env.variation_space["light"]["diffuse_scale"].low[0]) + 1e-9 or float(dark[o]) < 0.2
    ), f"{cell}: the declared dark-end gap does not exist"


# --------------------------------------------------------------------- dead upstream

#: Re-measuring ``dead_upstream`` means driving the LIVE swm wrapper through a variation reset, and
#: every reset that changes a model-level leaf makes swm recompile and drop the previous
#: ``dm_control`` ``Physics`` — which frees an EGL context on garbage collection. Releasing a context
#: corrupts every other live one in the process, which is exactly what ``stable_envs.testing.
#: resource_guard`` exists to catch, so this probe runs in a SUBPROCESS. That is not a workaround for
#: the guard: it is a finding about the swm wrapper, recorded in the research doc, and the isolation
#: is what makes the measurement safe to run beside the rest of the suite.
_DEAD_PROBE = r"""
import hashlib, json, sys
import numpy as np, gymnasium, stable_worldmodel  # noqa: F401
from tests.helpers.reference import variation_map
from tests.helpers.reference.variation_maps.dmc_control import MAPS
from stable_envs.registry import REGISTRY

def probe(leaf):
    iv = np.asarray(leaf.init_value)
    if hasattr(leaf, "n"):
        return int((int(iv) + 1) % int(leaf.n))
    lo = np.asarray(leaf.low, np.float64).reshape(np.shape(iv))
    hi = np.asarray(leaf.high, np.float64).reshape(np.shape(iv))
    far = np.where(np.abs(iv - lo) > np.abs(iv - hi), lo, hi)
    return np.asarray(iv + 0.9 * (far - iv), leaf.dtype)

out = {}
for cell, m in MAPS.items():
    if not m.dead_upstream:
        continue
    size = int(getattr(REGISTRY[cell], "default_size", 84))
    env = gymnasium.make(m.swm_id)
    h = env
    while getattr(h, "variation_space", None) is None:
        h = h.env
    def digest():
        mod = h.env.physics.model
        mod = mod.ptr if hasattr(mod, "ptr") else mod
        names = sorted(a for a in dir(mod) if not a.startswith("_")
                       and isinstance(getattr(mod, a, None), np.ndarray))
        blob = b"|".join(np.ascontiguousarray(np.asarray(getattr(mod, a))).tobytes() for a in names)
        return hashlib.sha256(blob + np.asarray(mod.opt.gravity).tobytes()).hexdigest()
    def frames():
        fr = []
        for k in range(4):
            h.reset(seed=200 + k, options=dict(opts))
            fr.append(h.render(width=size, height=size, camera_id=h.camera_id).copy())
        return np.stack(fr)
    opts = {}
    h.reset(seed=5)
    base_model = digest()
    base_px = frames()
    rec = {}
    for path in m.dead_upstream:
        leaf = h.variation_space
        for part in path.split("."):
            leaf = leaf[part]
        opts = {"variation_values": {path: probe(leaf)}}
        h.reset(seed=5, options=dict(opts))
        rec[path] = {"model_unchanged": digest() == base_model,
                     "px_changed": int((np.abs(frames().astype(np.int32)
                                               - base_px.astype(np.int32)).max(-1) > 0).sum())}
        opts = {}
        h.reset(seed=5)
    out[cell] = rec
    env.close()
json.dump(out, open(sys.argv[1], "w"))
"""


@needs_swm
def test_dead_upstream_leaves_really_are_dead(tmp_path):
    """Every ``dead_upstream`` leaf, re-measured against live swm, in one isolated subprocess.

    Two mechanisms are allowed, and a leaf must satisfy one of them:

    * it changes no array of the compiled ``MjModel`` at all (the wrapper never reads it, or MuJoCo
      ignores the write because the geom carries an explicit mass);
    * it changes the model and no pixel, because the element it reaches is in a geom/site group
      ``mjvOption`` does not draw — acrobot's group-3 target and ball-in-cup's group-4 one.

    A leaf that moves both is not dead and the classification is wrong.
    """
    import json
    import subprocess
    import sys

    if sys.platform == "darwin":
        pytest.skip(
            "this probe needs dm_control physics.render; on macOS that either hangs in GLFW or is "
            "bound to MUJOCO_GL=disable so import of stable_envs cannot hang"
        )

    script = tmp_path / "probe.py"
    script.write_text(_DEAD_PROBE)
    result = tmp_path / "out.json"
    import os

    import stable_envs

    # the subprocess must import the SAME checkout pytest is running against; the repo is normally on
    # sys.path via the root conftest rather than installed.
    root = os.path.dirname(os.path.dirname(os.path.abspath(stable_envs.__file__)))
    env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, str(script), str(result)], capture_output=True, text=True, timeout=900, env=env
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    measured = json.loads(result.read_text())

    declared = {c: set(variation_map(c).dead_upstream) for c in CELLS if variation_map(c).dead_upstream}
    assert set(measured) == set(declared)
    for cell, leaves in measured.items():
        assert set(leaves) == declared[cell], cell
        for path, got in leaves.items():
            assert got["model_unchanged"] or got["px_changed"] == 0, (
                f"{cell}: {path} is declared dead upstream but moves the model AND {got['px_changed']} pixels"
            )


# --------------------------------------------------------------------- backward compatibility


@pytest.mark.parametrize("cell", ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "PushT"])
def test_the_four_pre_step8_maps_are_untouched(cell):
    """The extra mapping fields are additive: the maps that predate them behave exactly as they did.

    ``unmapped()`` must equal ``swm_only`` plus measured ``dead_upstream`` and ``mapped_swm_paths()`` the old
    ``{t for t, _ in pairs().values()}``, because those two expressions are what the old
    ``style_from_variation``/``check_mapping`` used.
    """
    m = VARIATION_MAPS[cell]
    assert m.unmapped() == {**m.swm_only, **m.dead_upstream}
    assert m.mapped_swm_paths() == {t for t, _ in m.pairs().values()}
    assert not m.rgb255 and not m.enum and not m.affine and not m.unsupported
    assert not m.not_applicable
    # ``affine_pairs`` is the general view and must agree with ``pairs`` when ``affine`` is empty
    assert m.affine_pairs() == {o: (t, f, 0.0) for o, (t, f) in m.pairs().items()}


@needs_dmc
def test_enum_alternatives_round_trip():
    """The one ``enum`` leaf in this family: swm's Discrete(2) into our Discrete(3), both ways."""
    env = make("DMCManipulatorBringBall", num_envs=1, device="cpu")
    o, _ = _layout(env)["target_ball.shape"]
    for swm_index in (0, 1):
        row = style_from_variation(env, {"target.shape": swm_index})
        assert int(row[o]) == swm_index
        assert variation_from_style(env, row)["target.shape"] == swm_index
    # our third alternative is the compiled model's own and is what the DEFAULT style carries
    assert int(np.asarray(env.default_style()).reshape(-1)[o]) == 2


@needs_dmc
def test_an_unreachable_enum_index_is_refused():
    """A swm index the table does not cover must raise rather than silently pick something."""
    env = make("DMCManipulatorBringBall", num_envs=1, device="cpu")
    with pytest.raises(KeyError, match="no entry in the map"):
        style_from_variation(env, {"target.shape": 7})


def test_affine_is_invertible_and_additive():
    """The ``affine`` form: ``ours = theirs * factor + offset``.

    Exercised on a synthetic map rather than on one of this family's — nothing here needs an offset —
    so that the arithmetic is tested rather than one caller's use of it. The inverted-switch case is
    the invert: swm 0/1 into our 1/0.
    """
    from dataclasses import replace

    env = make("Pendulum", num_envs=1, device="cpu")
    m = replace(
        variation_map(env),
        identical=(),
        partial={},
        rgb255={},
        scaled={},
        affine={"rendering.antialias": ("visuals.bg", -1.0, 1.0)},
    )
    VARIATION_MAPS["_AffineProbe"] = m
    try:

        class Probe(type(env)):
            pass

        Probe.__name__ = "_AffineProbe"
        probe = Probe.__new__(Probe)
        probe.__dict__.update(env.__dict__)
        o, _ = _layout(env)["rendering.antialias"]
        for theirs, ours in ((0.0, 1.0), (1.0, 0.0), (0.25, 0.75)):
            row = style_from_variation(probe, {"visuals.bg": np.full(3, theirs, np.float32)})
            assert float(row[o]) == pytest.approx(ours)
            assert variation_from_style(probe, row)["visuals.bg"] == pytest.approx(theirs)
        assert m.mapped_swm_paths() == {"visuals.bg"}
        assert m.affine_pairs()["rendering.antialias"] == ("visuals.bg", -1.0, 1.0)
    finally:
        VARIATION_MAPS.pop("_AffineProbe")


def test_a_non_invertible_affine_is_refused():
    """``factor = 0`` collapses every swm value onto one of ours and cannot round-trip."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="not invertible"):
        replace(VARIATION_MAPS["Pendulum"], affine={"a.b": ("c.d", 0.0, 1.0)})


@needs_swm
@pytest.mark.parametrize("cell", CELLS)
def test_strict_accepts_the_whole_swm_space(cell):
    """``strict=True`` is the caller-facing form of the completeness claim."""
    env = _env(cell)
    style_from_variation(env, swm_defaults(variation_map(env).swm_id), strict=True)


# --------------------------------------------------------------- the capability patch


@needs_dmc
def test_the_parity_floor_set_is_the_plate_baking_tier():
    """``_PARITY_FLOOR`` is transcribed, so derive it from the classes and assert it.

    The set decides which mechanism each cell's ``floor.color`` note describes, and the module cannot
    import ``stable_envs.envs`` to derive it. A cell that moved between the two DM Control tiers
    would otherwise leave the map describing a renderer it no longer has.
    """
    from stable_envs.envs.dmcontrol import DMControlSuite
    from tests.helpers.reference.variation_maps.dmc_control import _PARITY_FLOOR

    derived = {c for c in MAPS if c in REGISTRY and issubclass(REGISTRY[c], DMControlSuite)}
    assert derived == set(_PARITY_FLOOR), (
        f"the plate-baking tier is {sorted(derived)} but the map declares {sorted(_PARITY_FLOOR)}"
    )


@needs_dmc
@pytest.mark.parametrize(
    "cell", sorted(c for c in MAPS if c in REGISTRY and MAPS[c].classification().get("floor.color") == "mapped")
)
def test_floor_color_is_a_leaf_and_the_default_is_the_identity(cell):
    """The two halves of "mapped": the leaf exists with swm's shape and default, and it moves pixels.

    The default half is the one that matters for certification and is the one a classification table
    cannot check for itself: swm's ``init_value`` for this leaf is dm_control's own
    ``common/materials.xml`` grid texture, so a correct implementation renders the registered default
    BYTE-IDENTICALLY to the pre-patch frame -- the leaf must add reach without moving the picture.
    """
    import sys

    from stable_envs.envs.dmcontrol import DMControlSuite

    if sys.platform == "darwin" and not issubclass(REGISTRY[cell], DMControlSuite):
        pytest.skip("the trackcom CPU scene caches its sky through dm_control physics.render, disabled on macOS")
    env = make(cell, num_envs=2, size=84, device="cpu")
    layout = dict((p, (o, w)) for p, o, w in env.variation_space.flat_layout())
    assert "floor.color" in layout and layout["floor.color"][1] == 6, layout
    d = np.asarray(env.variation_space["floor"]["color"].value, np.float64).reshape(2, 3)
    assert np.allclose(d, [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), d
    st = env.sample_states(2, np.random.default_rng(0))
    a = env._render(st)
    b = env._render(st, variation={"floor.color": np.asarray([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])})
    assert int((a != b).any(-1).sum()) > 0, f"{cell}: floor.color moved no pixel"
    c = env._render(st, variation={"floor.color": d})
    assert torch.equal(a, c), f"{cell}: rendering AT the declared default is not the default frame"
    assert torch.equal(a, env._render(st)), f"{cell}: a custom floor leaked into the next render"
