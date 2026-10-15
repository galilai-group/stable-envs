"""The eighteen DM Control cells: package surface, declarations, and the Env contract on CPU."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch

import stable_envs as snv

pytestmark = pytest.mark.unit

dmc = pytest.importorskip(
    "stable_envs.envs.dmcontrol", reason="the DM Control family needs dm_control and a GL backend"
)

# Written out rather than read off the package, so a cell that disappears fails here.
PARITY = (
    "DMCAcrobotBalance",
    "DMCBallInCupCatch",
    "DMCCartpoleBalance",
    "DMCCartpoleSwingup",
    "DMCFingerSpin",
    "DMCFingerTurnEasy",
    "DMCFingerTurnHard",
    "DMCManipulatorBringBall",
    "DMCPendulumSwingup",
    "DMCPointMassEasy",
    "DMCReacherEasy",
    "DMCReacherHard",
)
TRACKCOM = (
    "DMCCheetahRun",
    "DMCHopperHop",
    "DMCHumanoidWalk",
    "DMCQuadrupedWalk",
    "DMCSwimmerSwimmer6",
    "DMCWalkerWalk",
)
ALL = PARITY + TRACKCOM
CONTINUATION = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk")

WRITTEN_IN = {
    "DMCAcrobotBalance": "acrobot",
    "DMCBallInCupCatch": "ball_in_cup",
    "DMCCartpoleBalance": "cartpole",
    "DMCCartpoleSwingup": "cartpole",
    "DMCFingerSpin": "finger",
    "DMCFingerTurnEasy": "finger",
    "DMCFingerTurnHard": "finger",
    "DMCManipulatorBringBall": "manipulator",
    "DMCPendulumSwingup": "pendulum",
    "DMCPointMassEasy": "point_mass",
    "DMCReacherEasy": "reacher",
    "DMCReacherHard": "reacher",
    "DMCCheetahRun": "cheetah",
    "DMCHopperHop": "hopper",
    "DMCHumanoidWalk": "humanoid",
    "DMCQuadrupedWalk": "quadruped",
    "DMCSwimmerSwimmer6": "swimmer",
    "DMCWalkerWalk": "walker",
}


def _build(name, **kw):
    cls = snv.registry.REGISTRY[name]
    try:
        return cls(**kw)
    except Exception as exc:  # noqa: BLE001 — no model, no EGL, no GPU
        pytest.skip(f"{name} is not constructible here: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------- package surface


def test_package_exports_the_names_callers_import():
    assert hasattr(dmc, "__path__")
    for name in (
        "DOMAINS",
        "TRACKCOM_DOMAINS",
        "VISUAL_FIDELITY",
        "DEFAULT_SIZES",
        "CPU_UNCERTIFIED",
        "TRACKCOM_BASES",
        "SHAPE_KNOB_SPECS",
        "RENDER_INVARIANT_DIMS",
        "STATE_DIMS_NOTE",
        "ALL_EXACT_FLOOR",
        "PARITY_NOTE",
        "VISUAL_NOTE",
        "SPECULAR_NOTE",
        "QUADRUPED_NOTE",
        "TRACKCOM_RENDER_FLAGS",
        "DMControlSuite",
        "DMControlTrackCom",
        "_DMCModel",
        "_TranscribedQuadrupedModel",
        "_transcribed_quadruped_env",
        "_tc_context",
        "_TCtx",
        "_episode_slots",
        "_shade",
        "catenary",
        "_chain_capsules_batch",
        "_MJMINVAL",
        "_solve_catenary",
        "_ENV_CACHE",
        "_TC_CACHE",
        "_SCENE_CACHE",
        "SPECULAR_SCALE",
    ):
        assert hasattr(dmc, name), f"stable_envs.envs.dmcontrol lacks {name}"


@pytest.mark.parametrize("name", ALL)
def test_cell_is_registered_under_the_package_and_declared_in_its_domain_module(name):
    cls = snv.registry.REGISTRY[name]
    assert cls.__module__ == "stable_envs.envs.dmcontrol"
    assert getattr(dmc, name) is cls
    assert snv.info(name)["family"] == "dmcontrol"
    assert name in snv.list_envs(family="dmcontrol")
    domain_module = importlib.import_module(f"stable_envs.envs.dmcontrol.{WRITTEN_IN[name]}")
    assert getattr(domain_module, name) is cls


def test_package_tables_describe_exactly_the_registered_cells():
    assert set(dmc.DOMAINS) == set(PARITY) and set(dmc.TRACKCOM_DOMAINS) == set(TRACKCOM)
    assert dmc.DEFAULT_SIZES == {"DMCManipulatorBringBall": 224}
    assert snv.registry.REGISTRY["DMCManipulatorBringBall"].default_size == 224
    assert not dmc.CPU_UNCERTIFIED
    assert set(dmc.VISUAL_FIDELITY) == set(TRACKCOM)
    assert set(dmc.TRACKCOM_BASES) == {"DMCQuadrupedWalk", "DMCSwimmerSwimmer6", "DMCHumanoidWalk", *CONTINUATION}
    from stable_envs.envs.dmcontrol._aligned import DMCAlignedSemantics
    from stable_envs.envs.dmcontrol._continuation import DMCContinuationSemantics

    assert dmc.TRACKCOM_BASES["DMCQuadrupedWalk"] == (DMCAlignedSemantics, dmc._TranscribedQuadrupedModel)
    for name in CONTINUATION:
        assert dmc.TRACKCOM_BASES[name] == (DMCContinuationSemantics,)


@pytest.mark.parametrize("name", ALL)
def test_cell_is_a_declaration_on_its_renderer_tier(name):
    cls = snv.registry.REGISTRY[name]
    tier = dmc.DMControlSuite if name in PARITY else dmc.DMControlTrackCom
    assert issubclass(cls, tier)
    own = {k: v for k, v in vars(cls).items() if not k.startswith("__")}
    overridden = sorted(
        k for k, v in own.items() if inspect.isfunction(v) or isinstance(v, (classmethod, staticmethod))
    )
    assert not overridden, f"{name} overrides {overridden}; a cell should be a declaration"


def test_renderer_tiers_share_one_base():
    assert dmc.DMControlSuite.__bases__ == (dmc._DMCModel,)
    assert dmc.DMControlTrackCom.__bases__ == (dmc._DMCModel,)
    assert dmc._DMCModel.__bases__ == (snv.Env,)


def test_declared_provenance_floors_notes_and_cameras():
    for n in TRACKCOM:
        vf = snv.registry.REGISTRY[n].visual_fidelity
        assert vf["reference_versions"] == {"mujoco": "3.12.0"}, n
        assert vf["measured_with"] == {"mujoco_warp": "3.12.0"}, n
        assert "3.12.0" in vf["measured_at"], n
    assert dmc.ALL_EXACT_FLOOR == {"DMCFingerTurnEasy": 0.89}
    assert snv.registry.REGISTRY["DMCFingerTurnEasy"].parity_all_exact_min == 0.89
    interior = snv.registry.REGISTRY["DMCCartpoleBalance"].parity_interior_min
    assert all(snv.registry.REGISTRY[n].parity_interior_min == interior for n in PARITY)
    assert dmc.RENDER_INVARIANT_DIMS == {
        "DMCManipulatorBringBall": ("ball_y", "target_ball_quat_qw", "target_ball_quat_qy")
    }
    for n in ALL:
        want = dmc.RENDER_INVARIANT_DIMS.get(n, ())
        assert tuple(getattr(snv.registry.REGISTRY[n], "RENDER_INVARIANT_DIMS", ()) or ()) == want, n
    assert dmc.TRACKCOM_RENDER_FLAGS == {"DMCSwimmerSwimmer6": {"enable_specular": False}}
    assert dmc.SPECULAR_NOTE in snv.registry.REGISTRY["DMCSwimmerSwimmer6"].visual_note
    assert dmc.QUADRUPED_NOTE in snv.registry.REGISTRY["DMCQuadrupedWalk"].visual_note
    for n in TRACKCOM:
        assert snv.registry.REGISTRY[n].visual_note.startswith(dmc.VISUAL_NOTE)
    assert {n: snv.registry.REGISTRY[n].CAMERA for n in ALL if snv.registry.REGISTRY[n].CAMERA} == {
        "DMCQuadrupedWalk": 2
    }


def test_renderer_and_scene_caches_are_the_module_level_objects():
    """One ``mujoco.Renderer`` per (domain, size), never closed: releasing an EGL context corrupts the others."""
    from stable_envs.envs.dmcontrol import _model, suite, trackcom

    assert dmc._ENV_CACHE is _model._ENV_CACHE
    assert suite._SCENE_CACHE is _model._SCENE_CACHE
    assert trackcom._TC_CACHE is _model._TC_CACHE
    assert "_close" not in vars(dmc.DMControlSuite) and "_close" not in vars(dmc.DMControlTrackCom)


@pytest.mark.parametrize("name", CONTINUATION)
def test_trackcom_render_model_carries_the_swm_default_material(name):
    source = getattr(dmc, name)._load_env(17)
    try:
        rgba = source.physics.named.model.mat_rgba["self"]
        np.testing.assert_array_equal(rgba, np.asarray([0.7, 0.5, 0.3, 1.0], np.float32))
    finally:
        source.close()


# --------------------------------------------------------------- the Env contract, live


@pytest.mark.parametrize("name", ALL)
def test_constructor_len_and_idempotent_close(name):
    env = _build(name, num_envs=2, device="cpu")
    assert env.num_envs == 2 and len(env) == 2 and env.B == 2
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env._render()


@pytest.mark.parametrize("name", ALL)
def test_private_projection_roundtrips_the_renderer_layout_and_is_not_a_state(name):
    with _build(name, num_envs=4, device="cpu") as env:
        env._set_render_state(env.sample_states(4, np.random.default_rng(0)))
        rows = env._get_render_state()
        projection = env._unpack_render_state(rows)
        state = env.get_state()
        assert "qvel" in state
        assert env._pack_render_state(state).shape == rows.shape
        assert any(off is None for _, _, off in env._layout())
        with pytest.raises((TypeError, ValueError)):
            env.set_state(projection)


@pytest.mark.parametrize("name", ALL)
def test_render_of_state_packed_row_and_live_state_agree_to_the_bit(name):
    with _build(name, num_envs=2, device="cpu") as env:
        st = env.get_state()
        rows = env._pack_render_state(st)
        env._set_render_state(rows)
        a = env.render(st)
        b, c = env._render(rows), env._render()
        assert torch.equal(a, b) and torch.equal(a, c)
        assert tuple(a.shape) == (2, env.size, env.size, 3) and a.dtype == torch.uint8
        assert tuple(env.frame_spec.shape) == (env.size, env.size, 3)
