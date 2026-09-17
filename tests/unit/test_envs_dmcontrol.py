"""Workstream B's eighteen DM Control cells on the Wave-0 foundation, and the package they now live in.

The oracle that the BEHAVIOR did not change is ``scratch/step9/envs-dmcontrol/oracle.py`` (every
cell's corpus through both trees, byte-identical frames, identical declarations). This file pins two
things that oracle cannot:

  * the claims that are NEW in Step 9 and have no baseline to compare against — the ``num_envs=``
    constructor, the two-key public State, ``pack``/``unpack`` round-tripping the frozen packed row,
    the lifecycle, and the fact that a renderer-only cell REFUSES ``step``/``transition`` with the
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

dmc = pytest.importorskip("stable_envs.envs.dmcontrol",
                          reason="the DM Control family needs dm_control and a GL backend")

#: The twelve fixed-camera PARITY cells and the six ``trackcom`` VISUAL ones, written out rather than
#: read off the package: a test that derives its own expectation from the thing under test cannot see
#: a cell disappear.
PARITY = ("DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
          "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard", "DMCManipulatorBringBall",
          "DMCPendulumSwingup", "DMCPointMassEasy", "DMCReacherEasy", "DMCReacherHard")
TRACKCOM = ("DMCCheetahRun", "DMCHopperHop", "DMCHumanoidStand", "DMCQuadrupedWalk",
            "DMCSwimmerSwimmer6", "DMCWalkerWalk")
ALL = PARITY + TRACKCOM

#: Which domain module each cell is WRITTEN in. The registry reports the package for all eighteen
#: (see ``test_every_cell_reports_the_package_as_its_module``); this is the other half of that fact —
#: the cell is written where its domain is, which is what plan §3 asks for and what makes "what does
#: cheetah claim" a one-file question.
WRITTEN_IN = {
    "DMCAcrobotSwingup": "acrobot", "DMCBallInCupCatch": "ball_in_cup",
    "DMCCartpoleBalance": "cartpole", "DMCCartpoleSwingup": "cartpole",
    "DMCFingerSpin": "finger", "DMCFingerTurnEasy": "finger", "DMCFingerTurnHard": "finger",
    "DMCManipulatorBringBall": "manipulator", "DMCPendulumSwingup": "pendulum",
    "DMCPointMassEasy": "point_mass", "DMCReacherEasy": "reacher", "DMCReacherHard": "reacher",
    "DMCCheetahRun": "cheetah", "DMCHopperHop": "hopper", "DMCHumanoidStand": "humanoid",
    "DMCQuadrupedWalk": "quadruped", "DMCSwimmerSwimmer6": "swimmer", "DMCWalkerWalk": "walker",
}

#: Cells that refuse a device, and which. One cell, by name, with a reason — see ``swimmer.py``.
CPU_REFUSED = ("DMCSwimmerSwimmer6",)


def _build(name, **kw):
    """Construct a cell, or skip with the reason. A refusal this family DECLARES is not a skip."""
    cls = snv.REGISTRY[name]
    try:
        return cls(**kw)
    except ValueError:                           # a declared device refusal; the caller handles it
        raise
    except Exception as exc:                     # noqa: BLE001 — no model, no EGL, no GPU
        pytest.skip(f"{name} is not constructible here: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------- the shape of the migration

def test_the_import_path_is_the_package_and_it_carries_the_step8_names():
    """``stable_envs.envs.dmcontrol`` IS the package, and nothing that used the module lost a name.

    A module and a package cannot share a name, so unlike Workstream A's ``swm_nav`` merge there is
    no second path to alias — the package has to BE the old one. The names below are the ones the
    tests, the probes and ``stable_worldmodel`` actually reach for; the oracle checks all 79.
    """
    assert hasattr(dmc, "__path__"), "stable_envs.envs.dmcontrol should be the package"
    for name in ("DOMAINS", "TRACKCOM_DOMAINS", "VISUAL_FIDELITY", "DEFAULT_SIZES",
                 "CPU_UNCERTIFIED", "TRACKCOM_BASES", "SHAPE_KNOB_SPECS", "RENDER_INVARIANT_DIMS",
                 "STATE_DIMS_NOTE", "ALL_EXACT_FLOOR", "PARITY_NOTE", "VISUAL_NOTE",
                 "SPECULAR_NOTE", "QUADRUPED_NOTE", "TRACKCOM_RENDER_FLAGS",
                 "DMControlSuite", "DMControlTrackCom", "_DMCModel", "_SWMQuadrupedModel",
                 "_swm_quadruped_env", "_tc_context", "_TCtx", "_episode_slots", "_shade",
                 "catenary", "_chain_capsules_batch", "_MJMINVAL", "_solve_catenary",
                 "_ENV_CACHE", "_TC_CACHE", "_SCENE_CACHE", "MJW_CHUNK", "SPECULAR_SCALE"):
        assert hasattr(dmc, name), f"stable_envs.envs.dmcontrol lost {name}"


@pytest.mark.parametrize("name", ALL)
def test_every_cell_reports_the_package_as_its_module(name):
    """The registry row, the family and ``list_envs(family=...)`` did not move.

    ``registry.spec`` derives an env's FAMILY from the last component of ``__module__``, so letting
    ``type()`` default would have split one family of eighteen into fourteen named after domains and
    changed a column every Step-8 artifact recorded. The attribute is also ACCURATE: the class is
    reachable at that path, which the assertion checks rather than assumes.
    """
    cls = snv.REGISTRY[name]
    assert cls.__module__ == "stable_envs.envs.dmcontrol"
    assert getattr(dmc, name) is cls
    assert snv.spec(name)["family"] == "dmcontrol"
    assert name in snv.list_envs(family="dmcontrol")


@pytest.mark.parametrize("name", ALL)
def test_every_cell_is_written_in_its_own_domain_module(name):
    """Plan §3: one module per domain. The cell's DECLARATION lives where its domain does."""
    mod = importlib.import_module(f"stable_envs.envs.dmcontrol.{WRITTEN_IN[name]}")
    assert getattr(mod, name) is snv.REGISTRY[name]


def test_the_package_tables_are_views_of_the_cells_and_not_a_second_copy():
    """``DEFAULT_SIZES``, ``CPU_UNCERTIFIED``, ``VISUAL_FIDELITY`` and ``TRACKCOM_BASES`` are
    COLLECTED from the classes.

    That is the whole reason they can still exist: a table written beside the cells is a second place
    to change and the Step-8 module had four of them. Collected, the table cannot disagree with the
    class it describes — and this test is what says so, by comparing each entry against the class it
    was collected from rather than against a transcription.
    """
    assert dmc.DEFAULT_SIZES == {"DMCManipulatorBringBall": 224}
    assert snv.REGISTRY["DMCManipulatorBringBall"].default_size == 224
    assert set(dmc.CPU_UNCERTIFIED) == set(CPU_REFUSED)
    for n, reason in dmc.CPU_UNCERTIFIED.items():
        assert snv.REGISTRY[n].CPU_UNCERTIFIED_REASON == reason
    assert set(dmc.VISUAL_FIDELITY) == set(TRACKCOM)
    for n, vf in dmc.VISUAL_FIDELITY.items():
        assert snv.REGISTRY[n].visual_fidelity is vf
    assert set(dmc.TRACKCOM_BASES) == {"DMCQuadrupedWalk"}
    assert dmc.TRACKCOM_BASES["DMCQuadrupedWalk"] == (dmc._SWMQuadrupedModel,)
    assert set(dmc.DOMAINS) == set(PARITY) and set(dmc.TRACKCOM_DOMAINS) == set(TRACKCOM)


@pytest.mark.parametrize("name", ALL)
def test_a_cell_is_a_declaration_and_overrides_no_renderer_method(name):
    """No giant inheritance hierarchy (plan §12 B): a leaf carries DATA and inherits behaviour.

    Every one of the eighteen is exactly two rungs deep — ``_DMCModel`` -> one tier -> the leaf — and
    the leaf's own namespace holds nothing but declarations. A cell that needed to override a
    renderer method would be saying its domain does not belong in the tier it declared, and this is
    where that would be caught. ``DMCQuadrupedWalk`` carries one extra BASE, not an override: its
    model is ``stable_worldmodel``'s rather than ``suite.load``'s.
    """
    cls = snv.REGISTRY[name]
    tier = dmc.DMControlSuite if name in PARITY else dmc.DMControlTrackCom
    assert issubclass(cls, tier)
    own = {k: v for k, v in vars(cls).items() if not k.startswith("__")}
    callables = sorted(k for k, v in own.items()
                       if inspect.isfunction(v) or isinstance(v, (classmethod, staticmethod)))
    assert not callables, f"{name} overrides {callables}; a cell should be a declaration"
    extra_bases = tuple(b for b in cls.__bases__ if b is not tier)
    assert extra_bases == ((dmc._SWMQuadrupedModel,) if name == "DMCQuadrupedWalk" else ())


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
        vf = snv.REGISTRY[n].visual_fidelity
        assert vf["reference_versions"] == {"mujoco": "3.11.0"}, n
        assert vf["measured_with"] in ({"mujoco_warp": "3.11.0"}, {"warp": "1.16.0"}), n
        assert "3.11.0" in vf["measured_at"], n
    # the one declared all-pixel floor, and the interior floor NOT relaxed anywhere
    assert dmc.ALL_EXACT_FLOOR == {"DMCFingerTurnEasy": 0.89}
    assert snv.REGISTRY["DMCFingerTurnEasy"].parity_all_exact_min == 0.89
    for n in PARITY:
        assert snv.REGISTRY[n].parity_interior_min == snv.REGISTRY["DMCCartpoleBalance"].parity_interior_min
    # Manipulator's task-specific state exclusions, and nobody else's
    assert dmc.RENDER_INVARIANT_DIMS == {
        "DMCManipulatorBringBall": ("ball_y", "target_ball_quat_qw", "target_ball_quat_qy")}
    for n in ALL:
        want = dmc.RENDER_INVARIANT_DIMS.get(n, ())
        assert tuple(getattr(snv.REGISTRY[n], "RENDER_INVARIANT_DIMS", ()) or ()) == want, n
    # the trackcom flags, and swimmer's specular note reaching its visual_note
    assert dmc.TRACKCOM_RENDER_FLAGS == {"DMCSwimmerSwimmer6": {"enable_specular": False}}
    assert dmc.SPECULAR_NOTE in snv.REGISTRY["DMCSwimmerSwimmer6"].visual_note
    assert dmc.QUADRUPED_NOTE in snv.REGISTRY["DMCQuadrupedWalk"].visual_note
    for n in TRACKCOM:
        assert snv.REGISTRY[n].visual_note.startswith(dmc.VISUAL_NOTE)
    # the trackcom cameras: quadruped is the one cell not at camera 0
    assert {n: snv.REGISTRY[n].CAMERA for n in ALL if snv.REGISTRY[n].CAMERA} == {
        "DMCQuadrupedWalk": 2}


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


# --------------------------------------------------------------- the Step-9 contract, live

@pytest.mark.parametrize("name", ALL)
def test_takes_the_step9_constructor_and_the_lifecycle(name):
    """``num_envs=`` on EVERY cell (contract §2), ``len``, an idempotent ``close``, and a closed env
    that refuses to render rather than drawing through a backend the caller believes is gone."""
    if name in CPU_REFUSED:
        with pytest.raises(ValueError, match="CPU_UNCERTIFIED"):
            snv.REGISTRY[name](num_envs=2, device="cpu")
        return
    env = _build(name, num_envs=2, device="cpu")
    assert env.num_envs == 2 and len(env) == 2 and env.B == 2
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.render()


@pytest.mark.parametrize("name", ALL)
def test_the_public_state_is_qpos_plus_the_episode_draw_and_hides_nothing(name):
    """Contract §3, and the reason this family's ``state_layout`` is built per INSTANCE.

    ``qpos`` is the simulator's configuration; ``episode`` is what ``initialize_episode`` wrote into
    the MODEL, which is a different kind of quantity and is the whole subject of the widening. Every
    packed column belongs to one of the two, so no field has offset ``None``, ``_hidden_state`` is
    never consulted, and ``pack``/``unpack`` round-trip the frozen corpus layout bit for bit.
    """
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=4, device="cpu") as env:
        lay = env._layout()
        keys = [".".join(k) for k, _, _ in lay]
        assert keys in (["qpos"], ["qpos", "episode"]), keys
        assert all(off is not None for _, _, off in lay), "this family hides no field"
        assert sum(w for _, w, _ in lay) == env.state_dim
        assert lay[0][1] == env.nq
        env.set_state(env.sample_states(4, np.random.default_rng(0)))
        rows = env.get_state()
        assert torch.equal(env.pack(env.unpack(rows)), rows)
        assert torch.equal(env.pack(env.state), rows)
        assert env._hidden_state() == {} and env._live_state() == {}
        # the State keys name the packed columns they occupy, in order and covering all of them
        named = [n for k in keys for n in env.state_names_of(k)]
        assert named == list(env.state_names)


@pytest.mark.parametrize("name", ALL)
def test_a_renderer_only_cell_refuses_step_and_transition_and_says_why(name):
    """There is no batched simulator in this package for any DM Control cell, and that is DECLARED.

    Contract §5: an env without dynamics raises from ``step``/``transition`` with the exact reason,
    while ``reset`` and ``render`` always work. The alternative — inventing a simulator, or
    fabricating a zero reward — is what this test exists to prevent.
    """
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=2, device="cpu") as env:
        assert env.has_dynamics is False
        a = torch.zeros(2, 1, device=env.device)
        for call, args in (("step", (a,)), ("transition", (env.state, a))):
            with pytest.raises(NotImplementedError) as e:
                getattr(env, call)(*args)
            assert name in str(e.value) and "simulator" in str(e.value)
        assert env.reset(seed=0).batch_size[0] == 2


@pytest.mark.parametrize("name", ALL)
def test_render_takes_a_state_a_packed_row_or_nothing(name):
    """The frame is a function of the state, and the three ways of saying so agree to the bit."""
    if name in CPU_REFUSED:
        pytest.skip(f"{name} declares no CPU backend")
    with _build(name, num_envs=2, device="cpu") as env:
        env.set_state(env.sample_states(2, np.random.default_rng(0)))
        st = env.state
        a, b, c = env.render(st), env.render(env.pack(st)), env.render()
        assert torch.equal(a, b) and torch.equal(a, c)
        assert tuple(a.shape) == (2, env.size, env.size, 3) and a.dtype == torch.uint8
        assert tuple(env.observation_spec.shape) == (env.size, env.size, 3)
