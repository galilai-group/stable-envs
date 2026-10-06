"""The eighteen DM Control cells on the Env contract, and the package they now live in.

The oracle that the BEHAVIOR did not change is the certified DM Control corpus (every
cell's corpus through both trees, byte-identical frames, identical declarations). This file pins two
things that oracle cannot:

  * the claims that are new on the public API and have no baseline to compare against — the ``num_envs=``
    constructor, full State for recovered cells and private projections round-tripping frozen rows,
    the lifecycle, and the fact that a renderer-only cell REFUSES ``step``/``set_state`` with the
    contract's precise error instead of inventing a simulator;
  * the SHAPE of the migration itself — that the package is what ``stable_envs.envs.dmcontrol`` now
    means, that every cell reports that path as its module (so the family, the registry row and
    ``list_envs(family=...)`` did not move), that each cell is declared in its own domain module, and
    that the tables the package re-exports are VIEWS of the cells rather than a second copy that can
    drift.

The structural half needs no MuJoCo and runs everywhere. The behavioural half constructs cells, so it
is skipped where ``dm_control``/EGL is unavailable — which is the ordinary condition on a laptop and
is why the file is written in two halves rather than one.
"""

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

#: The twelve fixed-camera PARITY cells and the six ``trackcom`` VISUAL ones, written out rather than
#: read off the package: a test that derives its own expectation from the thing under test cannot see
#: a cell disappear.
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
ALIGNED = ("DMCBallInCupCatch", "DMCFingerTurnHard", "DMCManipulatorBringBall", "DMCQuadrupedWalk")
HUMANOID = ("DMCHumanoidWalk",)
RECOVERED = (
    ("DMCReacherHard", "DMCAcrobotBalance", "DMCPendulumSwingup", "DMCCartpoleSwingup")
    + CONTINUATION
    + ALIGNED
    + HUMANOID
)

#: Which domain module each cell is WRITTEN in. The registry reports the package for all eighteen
#: (see ``test_every_cell_reports_the_package_as_its_module``); this is the other half of that fact —
#: the cell is written where its domain is, which is what plan §3 asks for and what makes "what does
#: cheetah claim" a one-file question.
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

#: Cells that refuse a device, and which. One cell, by name, with a reason — see ``swimmer.py``.
CPU_REFUSED = ("DMCSwimmerSwimmer6",)


def _build(name, **kw):
    """Construct a cell, or skip with the reason. A refusal this family DECLARES is not a skip."""
    cls = snv.registry.REGISTRY[name]
    try:
        return cls(**kw)
    except ValueError:  # a declared device refusal; the caller handles it
        raise
    except Exception as exc:  # noqa: BLE001 — no model, no EGL, no GPU
        pytest.skip(f"{name} is not constructible here: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------- the shape of the migration


def test_the_import_path_is_the_package_and_it_carries_the_step8_names():
    """``stable_envs.envs.dmcontrol`` IS the package, and nothing that used the module lost a name.

    A module and a package cannot share a name, so unlike the ``swm_nav`` merge there is
    no second path to alias — the package has to BE the old one. The names below are the ones the
    tests, the probes and ``stable_worldmodel`` actually reach for; the oracle checks all 79.
    """
    assert hasattr(dmc, "__path__"), "stable_envs.envs.dmcontrol should be the package"
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
        "MJW_CHUNK",
        "SPECULAR_SCALE",
    ):
        assert hasattr(dmc, name), f"stable_envs.envs.dmcontrol lost {name}"


@pytest.mark.parametrize("name", ALL)
def test_every_cell_reports_the_package_as_its_module(name):
    """The registry row, the family and ``list_envs(family=...)`` did not move.

    Family is the explicit ``family`` class attribute. ``__module__`` is still pinned to the
    package so the class stays reachable at ``stable_envs.envs.dmcontrol``; letting ``type()``
    default would have named fourteen domain modules. The attribute is also ACCURATE: the class
    is reachable at that path, which the assertion checks rather than assumes.
    """
    cls = snv.registry.REGISTRY[name]
    assert cls.__module__ == "stable_envs.envs.dmcontrol"
    assert getattr(dmc, name) is cls
    assert snv.info(name)["family"] == "dmcontrol"
    assert name in snv.list_envs(family="dmcontrol")


@pytest.mark.parametrize("name", ALL)
def test_every_cell_is_written_in_its_own_domain_module(name):
    """Plan §3: one module per domain. The cell's DECLARATION lives where its domain does."""
    mod = importlib.import_module(f"stable_envs.envs.dmcontrol.{WRITTEN_IN[name]}")
    assert getattr(mod, name) is snv.registry.REGISTRY[name]


def test_the_package_tables_are_views_of_the_cells_and_not_a_second_copy():
    """``DEFAULT_SIZES``, ``CPU_UNCERTIFIED``, ``VISUAL_FIDELITY`` and ``TRACKCOM_BASES`` are
    COLLECTED from the classes.

    That is the whole reason they can still exist: a table written beside the cells is a second place
    to change and the package had four of them. Collected, the table cannot disagree with the
    class it describes — and this test is what says so, by comparing each entry against the class it
    was collected from rather than against a transcription.
    """
    assert dmc.DEFAULT_SIZES == {"DMCManipulatorBringBall": 224}
    assert snv.registry.REGISTRY["DMCManipulatorBringBall"].default_size == 224
    assert set(dmc.CPU_UNCERTIFIED) == set(CPU_REFUSED)
    for n, reason in dmc.CPU_UNCERTIFIED.items():
        assert snv.registry.REGISTRY[n].CPU_UNCERTIFIED_REASON == reason
    assert set(dmc.VISUAL_FIDELITY) == set(TRACKCOM)
    for n, vf in dmc.VISUAL_FIDELITY.items():
        assert snv.registry.REGISTRY[n].visual_fidelity is vf
    assert set(dmc.TRACKCOM_BASES) == {"DMCQuadrupedWalk", *CONTINUATION, *HUMANOID}
    from stable_envs.envs.dmcontrol._aligned import DMCAlignedSemantics

    assert dmc.TRACKCOM_BASES["DMCQuadrupedWalk"] == (DMCAlignedSemantics, dmc._TranscribedQuadrupedModel)
    from stable_envs.envs.dmcontrol._continuation import DMCContinuationSemantics

    for name in CONTINUATION:
        assert dmc.TRACKCOM_BASES[name] == (DMCContinuationSemantics,)
    assert set(dmc.DOMAINS) == set(PARITY) and set(dmc.TRACKCOM_DOMAINS) == set(TRACKCOM)


@pytest.mark.parametrize("name", ALL)
def test_a_cell_is_a_declaration_and_overrides_no_renderer_method(name):
    """No giant inheritance hierarchy (plan §12 B): a leaf carries DATA and inherits behaviour.

    Every one of the eighteen is exactly two rungs deep — ``_DMCModel`` -> one tier -> the leaf — and
    the leaf's own namespace holds nothing but declarations. A cell that needed to override a
    renderer method would be saying its domain does not belong in the tier it declared, and this is
    where that would be caught. ``DMCQuadrupedWalk`` carries one extra BASE, not an override: its
    model is the transcribed quadruped MJCF rather than ``suite.load``'s.
    """
    cls = snv.registry.REGISTRY[name]
    tier = dmc.DMControlSuite if name in PARITY else dmc.DMControlTrackCom
    assert issubclass(cls, tier)
    own = {k: v for k, v in vars(cls).items() if not k.startswith("__")}
    callables = sorted(
        k for k, v in own.items() if inspect.isfunction(v) or isinstance(v, (classmethod, staticmethod))
    )
    assert not callables, f"{name} overrides {callables}; a cell should be a declaration"
    extra_bases = tuple(b for b in cls.__bases__ if b is not tier)
    if name == "DMCReacherHard":
        from stable_envs.envs.dmcontrol.reacher import _ReacherSemantics

        expected = (_ReacherSemantics,)
    elif name in ALIGNED:
        from stable_envs.envs.dmcontrol._aligned import DMCAlignedSemantics

        expected = (DMCAlignedSemantics,)
        if name == "DMCQuadrupedWalk":
            expected += (dmc._TranscribedQuadrupedModel,)
    elif name in CONTINUATION:
        from stable_envs.envs.dmcontrol._continuation import DMCContinuationSemantics

        expected = (DMCContinuationSemantics,)
    elif name in HUMANOID:
        from stable_envs.envs.dmcontrol._humanoid import DMCHumanoidSemantics

        expected = (DMCHumanoidSemantics,)
    elif name in RECOVERED:
        from stable_envs.envs.dmcontrol._first_family import FirstFamilySemantics

        expected = (FirstFamilySemantics,)
    else:
        expected = (dmc._TranscribedQuadrupedModel,) if name == "DMCQuadrupedWalk" else ()
    assert extra_bases == expected


def test_the_two_tiers_are_the_only_thing_below_the_base():
    """One small base, two renderers. The hierarchy is three levels and stays three levels."""
    assert dmc.DMControlSuite.__bases__ == (dmc._DMCModel,)
    assert dmc.DMControlTrackCom.__bases__ == (dmc._DMCModel,)
    assert dmc._DMCModel.__bases__ == (snv.Env,)


def test_the_preserved_declarations_are_still_exactly_what_step_8_declared():
    """The five things plan §12 B names, asserted as literals rather than derived.

    Written out so that a change to any of them fails HERE, in a test whose name says what was
    changed, rather than only in the oracle's diff of two trees.
    """
    # mujoco 3.11 certification provenance: the declared arm is authoritative
    for n in TRACKCOM:
        vf = snv.registry.REGISTRY[n].visual_fidelity
        assert vf["reference_versions"] == {"mujoco": "3.11.0"}, n
        assert vf["measured_with"] in ({"mujoco_warp": "3.11.0"}, {"warp": "1.16.0"}), n
        assert "3.11.0" in vf["measured_at"], n
    # the one declared all-pixel floor, and the interior floor NOT relaxed anywhere
    assert dmc.ALL_EXACT_FLOOR == {"DMCFingerTurnEasy": 0.89}
    assert snv.registry.REGISTRY["DMCFingerTurnEasy"].parity_all_exact_min == 0.89
    for n in PARITY:
        assert (
            snv.registry.REGISTRY[n].parity_interior_min
            == snv.registry.REGISTRY["DMCCartpoleBalance"].parity_interior_min
        )
    # Manipulator's task-specific state exclusions, and nobody else's
    assert dmc.RENDER_INVARIANT_DIMS == {
        "DMCManipulatorBringBall": ("ball_y", "target_ball_quat_qw", "target_ball_quat_qy")
    }
    for n in ALL:
        want = dmc.RENDER_INVARIANT_DIMS.get(n, ())
        assert tuple(getattr(snv.registry.REGISTRY[n], "RENDER_INVARIANT_DIMS", ()) or ()) == want, n
    # the trackcom flags, and swimmer's specular note reaching its visual_note
    assert dmc.TRACKCOM_RENDER_FLAGS == {"DMCSwimmerSwimmer6": {"enable_specular": False}}
    assert dmc.SPECULAR_NOTE in snv.registry.REGISTRY["DMCSwimmerSwimmer6"].visual_note
    assert dmc.QUADRUPED_NOTE in snv.registry.REGISTRY["DMCQuadrupedWalk"].visual_note
    for n in TRACKCOM:
        assert snv.registry.REGISTRY[n].visual_note.startswith(dmc.VISUAL_NOTE)
    # the trackcom cameras: quadruped is the one cell not at camera 0
    assert {n: snv.registry.REGISTRY[n].CAMERA for n in ALL if snv.registry.REGISTRY[n].CAMERA} == {
        "DMCQuadrupedWalk": 2
    }


def test_the_one_renderer_per_domain_size_bundle_policy_is_still_a_module_level_cache():
    """The EGL rule: one ``mujoco.Renderer`` per (domain, size), built once and never closed.

    A cache that lived on a class or an instance would give a second env a second context, and
    releasing an EGL context corrupts every other live one in the process. The caches are therefore
    module-level in ``_model`` and every module that reads one reads THAT object — which is what the
    identity assertions below are for, since a ``from ... import`` of a dict shares the object while
    a re-declaration would not.
    """
    from stable_envs.envs.dmcontrol import _model, suite, trackcom

    assert dmc._ENV_CACHE is _model._ENV_CACHE
    assert suite._SCENE_CACHE is _model._SCENE_CACHE
    assert trackcom._TC_CACHE is _model._TC_CACHE
    assert "_close" not in vars(dmc.DMControlSuite) and "_close" not in vars(dmc.DMControlTrackCom)


# --------------------------------------------------------------- the Env contract, live


@pytest.mark.parametrize("name", ALL)
def test_takes_the_step9_constructor_and_the_lifecycle(name):
    """``num_envs=`` on EVERY cell (contract §2), ``len``, an idempotent ``close``, and a closed env
    that refuses to render rather than drawing through a backend the caller believes is gone."""
    if name in CPU_REFUSED:
        with pytest.raises(ValueError, match="CPU_UNCERTIFIED"):
            snv.registry.REGISTRY[name](num_envs=2, device="cpu")
        return
    env = _build(name, num_envs=2, device="cpu")
    assert env.num_envs == 2 and len(env) == 2 and env.B == 2
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env._render()


@pytest.mark.parametrize("name", ALL)
def test_private_projection_roundtrips_the_frozen_renderer_layout(name):
    """Compact renderer rows remain stable and never become public snapshots."""
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=4, device="cpu") as env:
        env._set_render_state(env.sample_states(4, np.random.default_rng(0)))
        rows = env._get_render_state()
        projection = env._unpack_render_state(rows)
        if name in RECOVERED:
            state = env.get_state()
            assert "qvel" in state
            assert env._pack_render_state(state).shape == rows.shape
            assert any(off is None for _, _, off in env._layout())
            with pytest.raises((TypeError, ValueError)):
                env.set_state(projection)
        else:
            assert torch.equal(env._pack_render_state(projection), rows)
            with pytest.raises(NotImplementedError, match="full simulator semantics"):
                env.get_state()


@pytest.mark.parametrize("name", [n for n in ALL if n not in RECOVERED])
def test_a_renderer_only_cell_refuses_public_simulator_operations(name):
    """Unrecovered cells explicitly refuse simulator State, reset and dynamics."""
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=2, device="cpu") as env:
        assert env.has_dynamics is False
        action = torch.zeros(2, 1, device=env.device)
        projection = env._unpack_render_state(env._get_render_state())
        for call, args in (
            ("get_state", ()),
            ("reset", ()),
            ("step", (action,)),
            ("set_state", (projection,)),
        ):
            with pytest.raises(NotImplementedError, match="full simulator semantics"):
                getattr(env, call)(*args)


@pytest.mark.parametrize("name", ALL)
def test_private_render_takes_a_projection_a_packed_row_or_nothing(name):
    """The frame is a function of the state, and the three ways of saying so agree to the bit."""
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=2, device="cpu") as env:
        if name in RECOVERED:
            st = env.get_state()
            rows = env._pack_render_state(st)
            env._set_render_state(rows)
            a = env.render(st)
        else:
            env._set_render_state(env.sample_states(2, np.random.default_rng(0)))
            rows = env._get_render_state()
            st = env._unpack_render_state(rows)
            a = env._render(st)
        b, c = env._render(rows), env._render()
        assert torch.equal(a, b) and torch.equal(a, c)
        assert tuple(a.shape) == (2, env.size, env.size, 3) and a.dtype == torch.uint8
        assert tuple(env.frame_spec.shape) == (env.size, env.size, 3)
