"""Craftax (Classic + Full) and PyFlytRocketLanding onto the Env contract.

Covers the constructor, the public :data:`State`, the packed render row, the renderer-only
refusals, the frame spec and the certified declarations. Renderer pixels are covered by
``test_craftax_atlas.py``, ``test_craftax_strip.py``, ``test_craftax_full.py`` and
``test_rocket_landing.py``.

These envs render stored states; Craftax/JAX and PyFlyt keep their own dynamics, so
``step``/``transition``/``unroll`` raise the contract's ``NotImplementedError``.

``PyFlytRocketLanding`` cannot be constructed without nvdiffrast, so its tests check the class and the
module: the constructor's signature, the CPU refusal's wording, and the declarations.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs._core import dynamics
from stable_envs.envs.craftax import classic as craftax
from stable_envs.envs.craftax import full as _craftax_full
from stable_envs.envs.rocket import legacy as RL

CRAFTAX = ("CraftaxClassic", "CraftaxFull")


@pytest.fixture(scope="module")
def envs():
    made = {name: getattr(snv, name)(num_envs=4, device="cpu") for name in CRAFTAX}
    yield made
    for e in made.values():
        e.close()


# ------------------------------------------------------------------ the `num_envs=` constructor


@pytest.mark.parametrize("name", CRAFTAX)
def test_takes_num_envs_and_device_and_closes(name):
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
    """Checked on the signature: the cell cannot be built without a CUDA host."""
    p = inspect.signature(RL.PyFlytRocketLanding.__init__).parameters
    assert list(p)[1] == "num_envs", "num_envs must be the first positional parameter"
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
    """Every packed column belongs to exactly one field, and no field has offset ``None``.

    Offset ``None`` marks state the packed row does not carry; a renderer-only env has none.
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
    """``_unpack_render_state`` returns storage the env does not later overwrite.

    ``.to(dtype)`` on a same-dtype tensor returns the same storage, and every field here is float32.
    """
    env = envs[name]
    a = torch.as_tensor(env.sample_states(4), dtype=torch.float32)
    b = torch.as_tensor(env.sample_states(4, np.random.default_rng(7)), dtype=torch.float32)
    env._set_render_state(a)
    st = env._unpack_render_state(env._get_render_state())
    snap = st["world", "blocks"].clone()
    env._set_render_state(b)
    assert torch.equal(st["world", "blocks"], snap)


def test_the_rocket_state_is_one_rigid_body_grouped_by_what_it_is():
    """Checked on the class: the cell cannot be built without a CUDA host."""
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
    # The offsets follow the frozen observation layout.
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
    st = env._unpack_render_state(env._get_render_state())
    px = env._render(st)
    assert torch.equal(px, env._render(rows))
    assert torch.equal(px, env._render())
    assert px.dtype == torch.uint8


@pytest.mark.parametrize("name,tiles", [("CraftaxClassic", (9, 9)), ("CraftaxFull", (13, 11))])
def test_the_frame_spec_is_the_frame_and_not_the_tile(envs, name, tiles):
    """``size`` is the tile edge for Craftax, so ``frame_shape`` and ``frame_spec`` give the whole frame."""
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
    """The renderer-only Craftax envs refuse ``step``, ``reset``, ``get_state`` and ``action_spec``."""
    env = envs[name]
    assert not env.has_dynamics and not dynamics.has_dynamics(env)
    a = torch.zeros(env.num_envs, 1)
    for call in (lambda: env.step(a, frames=None), env.reset, env.get_state, lambda: env.action_spec):
        with pytest.raises(NotImplementedError) as e:
            call()
        assert name in str(e.value) and "full simulator semantics" in str(e.value)
        assert "renders supplied states only" in str(e.value)


def test_the_rocket_refuses_a_cpu_device_by_design():
    """A ``device="cpu"`` request is refused, never served by CUDA.

    With nvdiffrast installed this raises ``NotImplementedError`` naming the missing CPU rasteriser;
    without it the ``ImportError`` for the unreachable renderer fires first. Both are accepted.
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


# ------------------------------------------------------------------ module layout


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


# ------------------------------------------------------------------ declarations


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


def test_frozen_craftax_classic_corpora_render():
    """The 64x127 world-row corpus is rejected until zero-extended to 149; the 64x149 corpus renders."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    world_rows = np.load(root / "tests/fixtures/states/fidelity/CraftaxClassic.npz")["states"]
    full_rows = np.load(root / "tests/fixtures/states/craftax_classic/states.npz")["states"]
    assert world_rows.shape == (64, 127) and full_rows.shape == (64, 149)
    env = snv.CraftaxClassic(num_envs=64, device="cpu")
    with pytest.raises(ValueError, match="expects state_dim 149, got 127"):
        env._render(torch.as_tensor(world_rows, dtype=torch.float32))
    wide = np.zeros((64, 149), np.float32)
    wide[:, :127] = world_rows
    wide[:, craftax.OBS_H * craftax.OBS_W * 2 + 2 + craftax.N_STRIP] = 1.0
    assert env._render(torch.as_tensor(wide)).shape == (64, 144, 144, 3)
    assert env._render(torch.as_tensor(full_rows, dtype=torch.float32)).shape == (64, 144, 144, 3)
    env.close()


def test_the_complete_observation_is_149_dimensional_and_the_strip_is_in_it():
    env = snv.CraftaxClassic(num_envs=2, device="cpu")
    assert env.state_dim == 149
    assert len([k for k, _, _ in env._layout() if k[0] == "inventory"]) == 16
    assert env.state_names[-4:] == ("rng_half0", "rng_half1", "rng_half2", "rng_half3")
    env.close()
