"""A step of one action without frames on a host-step family takes the short path with the same result."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch

import stable_envs as snv
from stable_envs._core import stepping
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

FAMILIES = [
    pytest.param("DMCReacher", {}, marks=pytest.mark.family("dmc")),
    pytest.param("DMCCartpoleSwingup", {}, marks=pytest.mark.family("dmc")),
    pytest.param("TwoRoom", {"simulation_backend": "numba"}, marks=pytest.mark.family("navigation")),
    pytest.param("FetchPush", {}, marks=pytest.mark.family("fetch")),
    pytest.param("FetchPickAndPlace", {}, marks=pytest.mark.family("fetch")),
    pytest.param("FetchReach", {}, marks=pytest.mark.family("fetch")),
    pytest.param("PointMazeMediumNavigate", {}, marks=pytest.mark.family("ogbench")),
    pytest.param("PointMazeTeleportNavigate", {}, marks=pytest.mark.family("ogbench")),
    pytest.param("HumanoidMazeMediumNavigate", {}, marks=pytest.mark.family("ogbench")),
]
#: Host steps of classic control and navigation; classic-control runs go past their time limit.
LONG_FAMILIES = [
    *(
        pytest.param(name, {"simulation_backend": "numba"}, marks=pytest.mark.family("navigation"), id=name)
        for name in ("TwoRoom", "TwoRoomTarget", "PiecewiseZones", "Piecewise")
    ),
    *(
        pytest.param(
            name, {"simulation_backend": backend}, marks=pytest.mark.family("classic"), id=f"{name}-{backend}"
        )
        for name, backend in (
            ("CartPole", "numba"),
            ("MountainCar", "numba"),
            ("Acrobot", "gym"),
            ("Pendulum", "numpy"),
            ("MountainCarContinuous", "gym"),
            ("MountainCarContinuous", "numpy"),
        )
    ),
]
CALLS = [
    {"states": None, "frames": None},
    {"frames": None},
    {"states": slice(1, None), "frames": None},
    {"frames": None, "state": False},
    {"states": None, "frames": None, "state": False},
]
LONG_CALLS = [*CALLS, {"states": slice(0, 0), "frames": None}]


def _same(actual, expected):
    assert list(actual.keys(True)) == list(expected.keys(True))
    assert_same_state(actual, expected)
    for key in actual.keys(True):
        node = actual[key]
        if not torch.is_tensor(node):
            assert node.batch_size == expected[key].batch_size, key


@pytest.mark.parametrize("name,kwargs", FAMILIES)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("sequence", [False, True])
def test_short_path_matches_the_general_path(name, kwargs, num_envs, sequence, monkeypatch):
    calls = []
    short = stepping._host_single_step

    def counted(env, *args):
        out = short(env, *args)
        if env is fast:
            calls.append(out is not None)
        return out

    monkeypatch.setattr(stepping, "_host_single_step", counted)
    fast = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    general = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    with fast, general:
        general.__dict__["_host_step"] = None
        fast.reset(seed=5)
        general.reset(seed=5)
        rng = np.random.default_rng(2)
        previous = None
        for call in CALLS * 2:
            action = fast.action_spec.sample(num_envs, rng=rng)
            if sequence:
                action = action.unsqueeze(1)
            a, b = fast.step(action, **call), general.step(action, **call)
            _same(a, b)
            if previous is not None:
                # A result owns its memory: later steps don't change it.
                _same(previous[0], previous[1])
            previous = (a, b.clone())
        _same(fast.get_state(), general.get_state())
    assert calls and all(calls)


def _time_limit(env) -> int | None:
    domain = getattr(env, "CLASSIC_DOMAIN", None)
    if domain is None:
        return None
    from stable_envs.envs.classic_control.simulation.batched import MAX_EPISODE_STEPS

    return MAX_EPISODE_STEPS[domain]


def compare_runs(fast, general, num_envs, sequence):
    """Step ``fast`` and ``general`` alike and compare every result, bit for bit.

    The calls cycle through ``LONG_CALLS``, and every fifth floating-point action is float16. The run restores a
    State and installs a new Variation at step 90, steps past a classic-control time limit (the restore keeps the
    elapsed steps) and checks that the episodes truncate, then resets again for its last 30 steps.
    """
    limit = _time_limit(fast)
    steps = (limit or 200) + 60
    fast.reset(seed=5)
    general.reset(seed=5)
    rng = np.random.default_rng(2)
    floating = fast.action_spec.dtype.is_floating_point
    previous = None
    truncated = False
    for t in range(steps):
        if t == 90:
            saved = general.get_state()
            variation = general.sample_variation(rng=np.random.default_rng(t))
            for env in (fast, general):
                env.set_state(saved)
                env.set_variation(variation)
        if t == steps - 30:
            fast.reset(seed=9)
            general.reset(seed=9)
        action = fast.action_spec.sample(num_envs, rng=rng).to(fast.device)
        if floating and t % 5 == 4:
            action = action.half()
        if sequence:
            action = action.unsqueeze(1)
        call = LONG_CALLS[t % len(LONG_CALLS)]
        a, b = fast.step(action, **call), general.step(action, **call)
        _same(a, b)
        if "truncated" in a.keys():
            truncated = truncated or bool(a["truncated"].any())
        if previous is not None:
            # A result owns its memory: later steps don't change it.
            _same(previous[0], previous[1])
        previous = (a, b.clone())
    _same(fast.get_state(), general.get_state())
    assert truncated or limit is None


@pytest.mark.parametrize("name,kwargs", LONG_FAMILIES)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("sequence", [False, True])
def test_long_host_runs_match_the_general_path(name, kwargs, num_envs, sequence, monkeypatch):
    calls = []
    short = stepping._host_single_step

    def counted(env, *args):
        out = short(env, *args)
        if env is fast:
            calls.append(out is not None)
        return out

    monkeypatch.setattr(stepping, "_host_single_step", counted)
    fast = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    general = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    with fast, general:
        general.__dict__["_host_step"] = None
        compare_runs(fast, general, num_envs, sequence)
    assert calls and all(calls)


NON_FINITE_CPU = [
    *((name, {}, True) for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "SimplePointMaze")),
    *(
        (name, {"simulation_backend": "torch"}, True)
        for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")
    ),
    # A JAX step doesn't wait for the device, so it doesn't check; get_state does.
    ("TwoRoom", {"simulation_backend": "jax"}, False),
]


@pytest.mark.family("navigation")
@pytest.mark.parametrize("name,kwargs,raises", NON_FINITE_CPU)
def test_a_cpu_step_to_a_non_finite_state_raises_except_with_jax(name, kwargs, raises):
    with snv.make(name, num_envs=2, device="cpu", **kwargs) as env:
        env.reset(seed=0)
        nan = torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])
        if raises:
            with pytest.raises(ValueError, match="non-finite|nonfinite|NaN"):
                env.step(nan, frames=None)
        else:
            env.step(nan, frames=None)
            with pytest.raises(ValueError, match="non-finite|nonfinite|NaN"):
                env.get_state()


@pytest.mark.family("dmc")
def test_short_path_runs_with_the_torch_threads_of_the_general_path(monkeypatch):
    with snv.make("DMCReacher", num_envs=2, device="cpu") as env:
        env.reset(seed=1)
        seen = []
        host_step = env._host_step

        def recorded(*args):
            seen.append(torch.get_num_threads())
            return host_step(*args)

        env.__dict__["_host_step"] = recorded
        # A family that steps on a native thread pool gets one intra-op thread during the step, on either path.
        env.__dict__["native_step_threads"] = True
        before = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            env.step(env.action_spec.sample(2, rng=np.random.default_rng(0)), states=None, frames=None)
            assert seen == [1]
            assert torch.get_num_threads() == 2
        finally:
            torch.set_num_threads(before)
