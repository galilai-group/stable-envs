"""The variation maps of the five classic-control and thirteen DM Control cells, against live swm.

The swm-dependent tests skip when ``stable_worldmodel`` is not importable; run them in a venv that
has it. Trackcom cells need a GL backend to build, so those cases run on a GPU host.
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
    LIGHT_RANGE,
    MAPS,
    swm_defaults,
)

needs_swm = pytest.mark.swm
#: The DM Control cells register only when dm_control and a GL backend are importable; the two
#: tests below are the only ones that need a DM Control cell and no swm.
needs_dmc = pytest.mark.skipif(
    "DMCManipulatorBringBall" not in REGISTRY, reason="the DM Control cells did not register (dm_control/MUJOCO_GL)"
)


def _gl(cells):
    """Trackcom cells build their dm_control source through a GL context."""
    from stable_envs.envs.dmcontrol import DMControlSuite

    return [
        pytest.param(c, marks=pytest.mark.gl)
        if c.startswith("DMC") and c in REGISTRY and not issubclass(REGISTRY[c], DMControlSuite)
        else c
        for c in cells
    ]


CELLS = sorted(MAPS)
#: Cells whose swm registered default differs in appearance from the compiled model's, with the
#: style leaves the mapping moves to reach it.
DEFAULT_REACHED_THROUGH = {
    "DMCBallInCupCatch": {"light.diffuse_scale"},
    "DMCFingerTurnHard": {"light.diffuse_scale"},
    "DMCManipulatorBringBall": {"light.diffuse_scale", "target_ball.shape", "visibility.target_ball"},
}


def _env(cell):
    return make(cell, num_envs=1, device="cpu")


def _layout(env):
    return {p: (o, w) for p, o, w in env.variation_space.flat_layout()}


def _swm_space(m):
    import gymnasium

    return swm_variation_space(gymnasium.make(m.swm_id))


# --------------------------------------------------------------------- completeness


@needs_swm
@pytest.mark.parametrize("cell", _gl(CELLS))
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
@pytest.mark.parametrize("cell", _gl(CELLS))
def test_classification_counts_match_the_declaration(cell):
    """The four-way counts this module claims, re-derived from the live swm space."""
    env = _env(cell)
    r = check_mapping(env)
    got = {"total": len(r["classification"])}
    for label in ("mapped", "unsupported", "dead_upstream", "not_applicable"):
        got[label] = sum(1 for v in r["classification"].values() if v == label)
    assert got == EXPECTED_COUNTS[variation_map(env).swm_id], cell


# --------------------------------------------------------------------- the registered default


@needs_swm
@pytest.mark.parametrize("cell", _gl(CELLS))
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
    """At the registered default, swm's frame is the official gymnasium frame; ours must equal it."""
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
@pytest.mark.parametrize("cell", _gl(CELLS))
def test_mapped_leaves_are_isolated(cell):
    """Moving one swm leaf moves only its own style slots; one-to-many leaves move every slot the map names.

    Asserted on the style row, where a wrong slot is distinguishable from two elements sharing a colour.
    """
    env = _env(cell)
    m = variation_map(env)
    defaults = swm_defaults(m.swm_id)
    base = style_from_variation(env, defaults)
    layout = _layout(env)
    space = _swm_space(m)
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
@pytest.mark.parametrize("cell", _gl(CELLS))
def test_round_trip(cell):
    """swm value -> style row -> swm value, at the default and at a probe on every mapped leaf."""
    env = _env(cell)
    m = variation_map(env)
    defaults = swm_defaults(m.swm_id)
    space = _swm_space(m)
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
                # 8-bit colours lose at most half a level (1/510) and must round-trip to the same uint8.
                assert np.allclose(got, want, atol=1.0 / 510.0), f"{cell}: {path} {got} != {want}"
                assert np.array_equal(
                    (got * 255.0 + 0.5).astype(np.uint8), (np.clip(want, 0, 1) * 255.0 + 0.5).astype(np.uint8)
                ), f"{cell}: {path} does not round-trip to the same uint8"
            else:
                # float32 is the style row's dtype.
                assert np.allclose(got, want, rtol=6e-8, atol=3e-7), f"{cell}: {path} {got} != {want}"


def test_rgb255_is_swm_quantisation():
    """``rgb255`` reproduces swm's ``uint8(v * 255 + 0.5)``; a bare factor of 255 truncates 0.3 to 76, not 77."""
    env = make("Pendulum", num_envs=1, device="cpu")
    layout = _layout(env)
    o, w = layout["rod.color"]
    for t in np.linspace(0.0, 1.0, 513):
        row = style_from_variation(env, {"visuals.rod": np.full(3, t, np.float32)})
        want = (np.float32(t) * 255.0 + 0.5).astype(np.uint8)
        got = np.asarray(row[o : o + w]).astype(np.uint8)
        assert (got == want).all(), f"t={t}: {got} != {want}"


@needs_swm
@pytest.mark.parametrize("cell", _gl(sorted(LIGHT_RANGE)))
def test_light_intensity_lands_on_the_model(cell):
    """swm writes intensity absolutely, ours scales the model's own diffuse: they agree over the reachable range."""
    env = _env(cell)
    base, lo, hi, _ = LIGHT_RANGE[cell]
    pristine = np.asarray(type(env)._pristine()["light_diffuse"], np.float64)
    assert np.allclose(pristine, base), f"{cell}: declared light base {base} is not the model's"
    o, w = _layout(env)["light.diffuse_scale"]
    for intensity in np.linspace(lo, hi, 25):
        row = style_from_variation(env, {"light.intensity": np.float32([intensity])})
        got = np.clip(pristine * float(row[o]), 0.0, 1.0)
        assert np.allclose(got, intensity, atol=3e-7), f"{cell}: {got} != {intensity}"
    # the declared dark-end gap: intensities below `lo` need a scale under the space's own minimum
    scale_low = float(env.variation_space["light"]["diffuse_scale"].low[0])
    at_lo = style_from_variation(env, {"light.intensity": np.float32([lo])})
    dark = style_from_variation(env, {"light.intensity": np.float32([lo / 2.0])})
    assert float(at_lo[o]) == pytest.approx(scale_low, rel=1e-6), cell
    assert float(dark[o]) < scale_low - 1e-6, f"{cell}: the declared dark-end gap does not exist"


# --------------------------------------------------------------------- dead upstream

#: Re-measuring ``dead_upstream`` drives the live swm wrapper through variation resets, each of which
#: recompiles and drops the previous ``Physics``, freeing a GL context that corrupts the others live in
#: the process. The probe therefore runs in a subprocess.
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
    """Every ``dead_upstream`` leaf changes no compiled array, or changes the model but no pixel.

    The second case is an element in a geom/site group ``mjvOption`` does not draw (acrobot's group-3
    target, ball-in-cup's group-4 one). A leaf that moves both model and pixels is not dead.
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

    # the subprocess must import the checkout under test, which is on sys.path via the root conftest.
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


# --------------------------------------------------------------------- enum and affine


@needs_dmc
def test_enum_alternatives_round_trip():
    """The one ``enum`` leaf in this family: swm's Discrete(2) into our Discrete(3), both ways."""
    env = make("DMCManipulatorBringBall", num_envs=1, device="cpu")
    o, _ = _layout(env)["target_ball.shape"]
    for swm_index in (0, 1):
        row = style_from_variation(env, {"target.shape": swm_index})
        assert int(row[o]) == swm_index
        assert variation_from_style(env, row)["target.shape"] == swm_index
    # the third alternative is the compiled model's own sphere and the default style's
    assert int(np.asarray(env.default_style()).reshape(-1)[o]) == 2


@needs_dmc
def test_an_unreachable_enum_index_is_refused():
    """A swm index the table does not cover must raise rather than silently pick something."""
    env = make("DMCManipulatorBringBall", num_envs=1, device="cpu")
    with pytest.raises(KeyError, match="no entry in the map"):
        style_from_variation(env, {"target.shape": 7})


def test_affine_is_invertible_and_additive():
    """``affine`` maps ``ours = theirs * factor + offset`` and inverts it; the inverted switch maps 0/1 to 1/0."""
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


# --------------------------------------------------------------------- floor.color


@needs_dmc
def test_the_parity_floor_set_is_the_plate_baking_tier():
    """``_PARITY_FLOOR`` is transcribed; it must name exactly the plate-baking ``DMControlSuite`` cells."""
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
    """``floor.color`` has swm's shape and default, moves pixels, and rendering at the default is the default frame."""
    import sys

    from stable_envs.envs.dmcontrol import DMControlSuite

    if sys.platform == "darwin" and not issubclass(REGISTRY[cell], DMControlSuite):
        pytest.skip("the trackcom CPU scene caches its sky through dm_control physics.render, disabled on macOS")
    env = make(cell, num_envs=2, size=84, device="cpu")
    layout = _layout(env)
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
