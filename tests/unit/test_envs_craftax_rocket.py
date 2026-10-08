"""Craftax (Classic + Full) and PyFlytRocketLanding onto the Env contract.

What these pin is the part of the Env contract that a family agent OWES — the constructor, the
public :data:`State`, the compact render row it projects to, the renderer-only refusal and the
frame spec — plus the certified declarations named by hand. They deliberately do NOT
re-assert what ``test_craftax_atlas.py``, ``test_craftax_strip.py``, ``test_craftax_full.py`` and
``test_rocket_landing.py`` already assert about the renderers themselves: a migration that broke a
sprite would fail those, and duplicating them here would only make two places to update.

The three cells currently render stored states: Craftax/JAX and PyFlyt keep their own dynamics.
``step``/``transition``/``unroll`` must raise the contract's precise ``NotImplementedError`` — not an
``AttributeError``, and not a fabricated one-step integrator invented for the sake of the API.

``PyFlytRocketLanding`` cannot be CONSTRUCTED on a machine without nvdiffrast, so its tests are about the
class and the module: the constructor's signature, the CPU refusal's wording, and the declarations.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs import _dynamics as dynamics
from stable_envs.envs import rocket_landing as RL
from stable_envs.envs.craftax import classic as craftax
from stable_envs.envs.craftax import full as _craftax_full
from tests.helpers.stepping import branch_step, branch_steps

CRAFTAX = ("CraftaxClassic", "CraftaxFull")


@pytest.fixture(scope="module")
def envs():
    made = {name: getattr(snv, name)(num_envs=4, device="cpu") for name in CRAFTAX}
    yield made
    for e in made.values():
        e.close()


# ------------------------------------------------------------------ the `num_envs=` constructor


@pytest.mark.parametrize("name", CRAFTAX)
def test_takes_the_step9_constructor(name):
    cls = getattr(snv, name)
    with cls(num_envs=5, device="cpu") as env:
        assert env.num_envs == 5 and len(env) == 5
        assert env.device == torch.device("cpu")
        assert not env.closed
        env.close()
        assert env.closed
    env.close()  # idempotent, and after __exit__ closed it
    with pytest.raises(RuntimeError, match="is closed"):
        env._render(env.sample_states(2))


def test_the_rocket_constructor_takes_num_envs():
    """Asserted from the SIGNATURE, because this cell cannot be built without a CUDA host."""
    p = inspect.signature(RL.PyFlytRocketLanding.__init__).parameters
    assert list(p)[1] == "num_envs", "num_envs must sit in the slot batch sat in (positional callers)"
    assert p["num_envs"].default == 1
    assert "batch" not in p
    assert p["device"].default is None


# ------------------------------------------------------------------ the public State


#: Written out rather than derived from ``state_layout``: a test that reads the same attribute the
#: code reads asserts only that the attribute exists.
EXPECTED_KEYS = {
    "CraftaxClassic": [("world", "blocks"), ("world", "mobs"), ("player", "facing")]
    + [("inventory", n) for n in craftax.STRIP_NAMES]
    + [("player", "is_sleeping"), ("world", "light_level"), ("world", "rng_key")],
    "CraftaxFull": [
        ("world", "blocks"),
        ("world", "items"),
        ("world", "mobs"),
        ("world", "projectiles"),
        ("world", "light_map"),
        ("player", "facing"),
        ("player", "is_sleeping"),
        ("player", "level"),
        ("world", "light_level"),
        ("world", "rng_key"),
    ]
    + [("inventory", n) for n in _craftax_full.STRIP_SLOTS],
}


@pytest.mark.parametrize("name", CRAFTAX)
def test_state_keys_are_the_declared_ones_and_cover_the_packed_row(envs, name):
    env = envs[name]
    assert [tuple(k) for k, _, _ in env._layout()] == EXPECTED_KEYS[name]
    rows = torch.as_tensor(env.sample_states(4), dtype=torch.float32)
    st = env._unpack_render_state(rows)
    assert st.batch_size[0] == 4
    assert torch.equal(env._pack_render_state(st), rows), "pack(unpack(rows)) must be the frozen row, bit for bit"


@pytest.mark.parametrize("name", CRAFTAX)
def test_the_packed_row_is_the_frozen_one_and_no_field_is_hidden(envs, name):
    """Every packed column belongs to exactly one field, and NO field has offset ``None``.

    Offset ``None`` means "real dynamical state the packed row does not carry" — PushT's agent
    velocity. A renderer-only cell has none by definition: there is no simulator to hold it. If one
    ever appeared here it would be a sign that a hidden buffer had been introduced, which is the thing
    that makes ``transition`` impure.
    """
    env = envs[name]
    lay = env._layout()
    assert sum(w for _, w, o in lay if o is not None) == env.state_dim
    assert all(o is not None for _, _, o in lay)
    assert env._hidden_state() == {} and env._live_state() == {}
    # the columns tile the row with no gap and no overlap, in order
    covered = sorted((o, o + w) for _, w, o in lay)
    assert covered[0][0] == 0 and covered[-1][1] == env.state_dim
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


@pytest.mark.parametrize("name", CRAFTAX)
def test_state_names_of_agrees_with_the_frozen_column_names(envs, name):
    env = envs[name]
    flat = [n for k, _, _ in env._layout() for n in env.state_names_of(k)]
    assert flat == list(env.state_names)


@pytest.mark.parametrize("name", CRAFTAX)
def test_a_state_the_caller_keeps_is_not_a_view_of_the_env(envs, name):
    """Foundation defect R8, pinned for these cells: ``unpack`` owns its output.

    ``.to(dtype)`` on a same-dtype tensor returns the SAME storage, so before the foundation's
    ``copy=True`` a State handed to a caller changed under their feet at the next ``set_state`` —
    silently, with nothing raised. Every field of these three cells is float32, so they are exactly
    the shape that was exposed.
    """
    env = envs[name]
    a = torch.as_tensor(env.sample_states(4), dtype=torch.float32)
    b = torch.as_tensor(env.sample_states(4, np.random.default_rng(7)), dtype=torch.float32)
    env._set_render_state(a)
    st = env.state
    snap = st["world", "blocks"].clone()
    env._set_render_state(b)
    assert torch.equal(st["world", "blocks"], snap)


def test_the_rocket_state_is_one_rigid_body_grouped_by_what_it_is():
    """Checked on the CLASS: the cell cannot be built without a CUDA host."""
    lay = RL.PyFlytRocketLanding.state_layout
    assert [tuple(k) for k, _, _ in lay] == [
        ("rocket", "angular_velocity"),
        ("rocket", "orientation"),
        ("rocket", "linear_velocity"),
        ("rocket", "position"),
    ]
    assert [w for _, w, _ in lay] == [3, 4, 3, 3]
    assert [o for _, _, o in lay] == [0, 3, 7, 10]
    assert sum(w for _, w, _ in lay) == RL.PyFlytRocketLanding.state_dim == 13
    # the offsets are the frozen obs layout, not a re-grouping of it
    n = RL.PyFlytRocketLanding.state_names
    assert n[0:3] == ("ang_vel_x", "ang_vel_y", "ang_vel_z")
    assert n[3:7] == ("quat_x", "quat_y", "quat_z", "quat_w")
    assert n[7:10] == ("lin_vel_x", "lin_vel_y", "lin_vel_z")
    assert n[10:13] == ("pos_x", "pos_y", "pos_z")


# ------------------------------------------------------------------ render and the observation spec


@pytest.mark.parametrize("name", CRAFTAX)
def test_render_takes_a_state_a_packed_row_or_nothing(envs, name):
    env = envs[name]
    rows = torch.as_tensor(env.sample_states(4), dtype=torch.float32)
    env._set_render_state(rows)
    st = env.state
    px = env._render(st)
    assert torch.equal(px, env._render(rows))
    assert torch.equal(px, env._render())
    assert px.dtype == torch.uint8


@pytest.mark.parametrize("name,tiles", [("CraftaxClassic", (9, 9)), ("CraftaxFull", (13, 11))])
def test_the_frame_spec_is_the_frame_and_not_the_tile(envs, name, tiles):
    """``size`` is the TILE edge for Craftax, so the base class's square-``size`` assumption is wrong.

    Left alone, ``frame_spec`` reported ``(16, 16, 3)`` for a 144x144 picture — a buffer 81x too
    small — and ``(10, 10, 3)`` for full Craftax's 130x110 one, which is the wrong shape as well as
    the wrong size. Fixed in the family (``frame_shape``), not in the base class, exactly as classic
    control's 400x600 was.
    """
    env = envs[name]
    rows = torch.as_tensor(env.sample_states(2), dtype=torch.float32)
    h, w = tiles[0] * env.size, tiles[1] * env.size
    assert env.frame_shape == (h, w)
    assert tuple(env.frame_spec.shape) == (h, w, 3)
    assert tuple(env._render(rows).shape) == (2, h, w, 3)
    assert env.frame_spec.dtype == torch.uint8


@pytest.mark.parametrize("name", CRAFTAX)
def test_the_state_spec_leaves_match_the_layout(envs, name):
    env = envs[name]
    spec = env.state_spec
    for key, w, _ in env._layout():
        node = spec
        for part in key:
            node = node[part]
        assert tuple(node.shape) == (w,)


# ------------------------------------------------------------------ renderer-only: the refusals


@pytest.mark.parametrize("name", CRAFTAX)
def test_these_cells_have_no_dynamics_and_say_so_precisely(envs, name):
    """The check the brief asked for, answered: stable-envs ships no batched Craftax simulator.

    Craftax's own step is ``jax.jit``-ed inside ``craftax``/``stable_worldmodel``; this package renders
    the states that step produces and never advances one. So the contract's refusal is the correct
    ``step``, and inventing a one-step integrator to satisfy the API would be a different simulator
    wearing the same name.
    """
    env = envs[name]
    assert not env.has_dynamics and not dynamics.has_dynamics(env)
    a = torch.zeros(env.num_envs, 1)
    st = env.state
    for call in (
        lambda: env.step(a, frames=None),
        lambda: branch_step(env, st, a),
        lambda: branch_steps(env, st, a.unsqueeze(0)),
        lambda: env.action_spec,
    ):
        with pytest.raises(NotImplementedError) as e:
            call()
        assert "ships a renderer and no batched simulator" in str(e.value)
        assert "reset() and render() work as usual" in str(e.value)


def test_the_rocket_refuses_a_cpu_device_by_design():
    """A ``device="cpu"`` request is never quietly served by CUDA (plan §2's device honesty).

    On a host WITH nvdiffrast the CPU request raises ``NotImplementedError`` naming the missing CPU
    textured-triangle rasteriser; on a host without it the earlier ``ImportError`` fires first, which
    is also correct — it says the renderer is unreachable rather than that the device is unsupported.
    Both are asserted so the test means something on either kind of machine.
    """
    with pytest.raises((NotImplementedError, ImportError)) as e:
        RL.PyFlytRocketLanding(num_envs=1, device="cpu")
    msg = str(e.value)
    if isinstance(e.value, NotImplementedError):
        assert "has no CPU backend" in msg and "never served" in msg
    else:
        assert "cannot render in this interpreter" in msg


def test_the_rocket_is_registered_exactly_when_it_can_render():
    assert ("PyFlytRocketLanding" in snv.registry.REGISTRY) == (RL.missing_backend() is None)


# ------------------------------------------------------------------ the module split


def test_craftax_full_registers_from_the_family_package():
    """Discovery registers the full-game implementation from the Craftax package."""
    assert snv.CraftaxFull is _craftax_full.CraftaxFull
    assert snv.CraftaxFull.__module__ == "stable_envs.envs.craftax.full"
    for n in (
        "N_BLOCK",
        "N_STRIP",
        "STRIP_SLOTS",
        "STRIP_CELLS",
        "OBS_H",
        "OBS_W",
        "light_night_and_sleep",
        "night_vignette",
        "transparent_mob_slots",
        "_tent",
    ):
        assert hasattr(_craftax_full, n)


def test_the_two_craftax_modules_could_not_have_been_merged():
    """The evidence for the brief's fallback, kept beside the choice rather than only in findings.

    35 module-level names are shared and 15 of them hold DIFFERENT values, so one module object cannot
    serve both spellings and a merge would have had to rename constants that tests import by name —
    which is not behaviour-neutral, the condition the brief attached to the merge.

    The count is ASSERTED here rather than only stated. It was prose before, and it drifted: the
    delivered findings and this docstring both claimed 21, counting eight names that are in fact
    identical in the two modules. A claim that carries a number and cannot fail is not evidence.
    """
    shared = {
        "N_BLOCK": (17, 37),
        "N_STRIP": (16, 41),
        "STRIP_H": (2, 4),
        "TILE_PX_AGENT": (7, 10),
        "N_COUNT": (10, 100),
    }
    for n, (classic, full) in shared.items():
        assert getattr(craftax, n) == classic
        assert getattr(_craftax_full, n) == full
    assert craftax.OBS_H != _craftax_full.OBS_H

    #: The 15 shared names that genuinely conflict: 9 constants, 3 functions whose bodies differ, and
    #: 3 same-named but distinct per-device caches. Pinned so that a future edit which makes the two
    #: modules mergeable has to come here and say so.
    CONFLICTING = {
        "N_BLOCK",
        "N_COUNT",
        "N_STRIP",
        "N_TILE",
        "OBS_H",
        "OBS_W",
        "REFERENCE_TILE_PX",
        "STRIP_H",
        "TILE_PX_AGENT",
        "_unpack",
        "night_vignette",
        "strip_geometry",
        "_CONST_CACHE",
        "_GRID_CACHE",
        "_VIGNETTE_CACHE",
    }
    names = {n for n in dir(craftax) if not n.startswith("__")} & {
        n for n in dir(_craftax_full) if not n.startswith("__")
    }
    assert len(names) == 35, f"shared name count moved to {len(names)}; update the module decision"

    def _differs(n):
        x, y = getattr(craftax, n), getattr(_craftax_full, n)
        if isinstance(x, types.FunctionType) and isinstance(y, types.FunctionType):
            return x.__code__.co_code != y.__code__.co_code
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return not (
                isinstance(x, torch.Tensor)
                and isinstance(y, torch.Tensor)
                and x.shape == y.shape
                and bool((x == y).all())
            )
        if isinstance(x, dict) and isinstance(y, dict):
            return x is not y
        if type(x) is not type(y):
            return True
        try:
            return not bool(x == y)
        except Exception:  # noqa: BLE001 -- unorderable objects: identity decides
            return x is not y

    assert {n for n in names if _differs(n)} == CONFLICTING


def test_a_missing_craftax_is_recorded_under_the_historical_module_key():
    """Dependency diagnostics retain their keys after moving into the family package."""
    errs = snv.import_errors()
    assert "_craftax_full" not in errs
    assert "_craftax_full_tex" not in errs


# --------------------------------------------------- the certified declarations plan §12 E names


def test_the_jit_wrapper_declaration_survives_on_both_craftax_cells():
    from tests.helpers.reference.craftax_jit import CRAFTAX_JIT_GAP

    for name in ("CraftaxClassic", "CraftaxFull"):
        g = CRAFTAX_JIT_GAP[name]
        assert g["swm_wrapper_jits"] == ["reset", "step"]
        assert g["certified_exact_frac"] == 1.0 and g["certified_max_abs_diff"] == 0
        assert g["max_abs_diff_swm_wrapper_vs_certified"] == 1
        assert "un-jitted" in g["certified_reference"]
    c = CRAFTAX_JIT_GAP["CraftaxClassic"]
    assert c["live_swm_exact_frac_7px"] == 0.997854
    assert c["live_swm_exact_frac_16px"] == 0.997225


def test_the_resolution_consistency_applicability_declarations_survive():
    c = snv.CraftaxClassic.resolution_consistency_applicability
    assert "light_level >= 0.5" in c and "light_level < 0.5" in c
    assert c["light_level < 0.5"].startswith("not_applicable")
    assert "0.1634" in c["light_level < 0.5"]
    f = snv.CraftaxFull.resolution_consistency_applicability
    assert f["all bands"].startswith("not_applicable")
    assert f["still_required"].startswith("exact parity")


def test_the_parity_claims_and_their_reference_versions_survive():
    for cls in (snv.CraftaxClassic, snv.CraftaxFull):
        assert cls.parity_all_exact_min == 1.0
        assert cls.parity_interior_min == 1.0
        assert cls.fidelity_tier == "parity" and cls.render_decoder == "renderer"
        assert cls.reference_versions == {"craftax": "1.6.1", "jax": "0.11.1"}
        assert "jax_threefry_partitionable" in cls.parity_note


def test_the_rocket_colour_exactness_declaration_survives_verbatim():
    d = RL.PyFlytRocketLanding.color_exactness_applicability
    assert d["leaves"] == (
        "pad.color",
        "rocket.body_color",
        "rocket.booster_color",
        "rocket.fin_color",
        "rocket.leg_color",
    )
    assert len(d["leaves"]) == 5
    assert d["max_shading_factor"] == 0.993567
    assert d["reachable_ceiling_for_the_probe_targets"] == {"200": 198, "33": 32, "17": 16}
    assert d["official_renderer_fails_the_same_check"] is True
    assert d["status"].startswith("not_applicable")


def test_the_rocket_tier_and_reference_versions_survive():
    cls = RL.PyFlytRocketLanding
    assert cls.fidelity_tier == "visual" and cls.parity_interior_min is None
    v = cls.visual_fidelity
    assert v["reference_versions"] == {"PyFlyt": "0.29.0", "pybullet": "3.2.7"}
    assert (v["mean_abs_max"], v["within8_min"], v["ssim_min"]) == (10.0, 0.91, 0.80)
    assert v["mean_abs"] == 7.71 and v["ssim"] == 0.854
    assert cls.default_size == 480 and cls.default_supersample == 1
    assert "NOT a substitute" in cls.visual_note


def test_the_rocket_flame_and_ignition_semantics_survive():
    assert RL.FLAME_RGBA == (1.0, 0.8, 0.0, 0.1)
    assert RL.IGNITION_RGBA == (1.0, 0.3, 0.3, 0.8)
    lay = RL.style_layout()
    assert lay["flame.ignited"] == (0, 1)
    assert set(RL.CORPUS_STYLE_PATHS) <= set(lay)
    assert RL.style_dim() == len(RL._default_style_row())


def test_the_batch_invariant_float64_mvp_does_not_depend_on_the_batch():
    """The camera is computed on the host in float64 and must not vary with how it is chunked."""
    rng = np.random.default_rng(0)
    st = np.asarray(rng.normal(size=(9, 13)), np.float64)
    st[:, 3:7] /= np.linalg.norm(st[:, 3:7], axis=1, keepdims=True)
    _, _, pos, q, _, _ = RL.lagged_pose(st)
    eye, cmat = RL.camera_pose(pos, q)
    assert eye.dtype == np.float64 and cmat.dtype == np.float64
    for lo, hi in ((0, 4), (4, 9), (2, 7)):
        _, _, p2, q2, _, _ = RL.lagged_pose(st[lo:hi])
        e2, c2 = RL.camera_pose(p2, q2)
        assert np.array_equal(e2, eye[lo:hi]) and np.array_equal(c2, cmat[lo:hi])


# ------------------------------------------------------------------ the frozen corpora


def test_both_frozen_craftax_classic_corpora_are_kept_and_both_replay():
    """Ruling: keep both. They are two different claims about the same cell.

    The 64x127 Step-4 array predates the nine-row world window, so it is the world-row oracle under
    the zero-extension interpretation, and rendering it DIRECTLY must raise rather than reshape. The
    64x149 Step-7 array certifies the complete observation.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    step4 = np.load(root / "tests/fixtures/states/fidelity/CraftaxClassic.npz")["states"]
    step7 = np.load(root / "tests/fixtures/states/craftax_classic/states.npz")["states"]
    assert step4.shape == (64, 127) and step7.shape == (64, 149)
    env = snv.CraftaxClassic(num_envs=64, device="cpu")
    with pytest.raises(ValueError, match="expects state_dim 149, got 127"):
        env._render(torch.as_tensor(step4, dtype=torch.float32))
    wide = np.zeros((64, 149), np.float32)
    wide[:, :127] = step4
    wide[:, craftax.OBS_H * craftax.OBS_W * 2 + 2 + craftax.N_STRIP] = 1.0
    assert env._render(torch.as_tensor(wide)).shape == (64, 144, 144, 3)
    assert env._render(torch.as_tensor(step7, dtype=torch.float32)).shape == (64, 144, 144, 3)
    env.close()


def test_the_complete_observation_is_149_dimensional_and_the_strip_is_in_it():
    env = snv.CraftaxClassic(num_envs=2, device="cpu")
    assert env.state_dim == 149
    assert len([k for k, _, _ in env._layout() if k[0] == "inventory"]) == 16
    assert env.state_names[-4:] == ("rng_half0", "rng_half1", "rng_half2", "rng_half3")
    env.close()
