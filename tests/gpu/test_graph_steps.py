"""A CUDA step of one action replayed as a CUDA graph gives the result of the step loop, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from _shared import make_or_skip  # noqa: E402

from stable_envs._core import stepping  # noqa: E402
from tests.unit.test_host_single_step import CALLS, _same  # noqa: E402

CLASSES = [
    *(pytest.param(name, {}, marks=pytest.mark.family("classic"), id=name)
      for name in ("CartPole", "Acrobot", "Pendulum", "MountainCar", "MountainCarContinuous")),
    *(pytest.param(name, {}, marks=pytest.mark.family("navigation"), id=name)
      for name in ("TwoRoomTarget", "PiecewiseZones", "Piecewise", "SimplePointMaze")),
    pytest.param("TwoRoom", {"simulation_backend": "torch"}, marks=pytest.mark.family("navigation"), id="TwoRoom"),
]  # fmt: skip


@pytest.mark.parametrize("name,kwargs", CLASSES)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("sequence", [False, True])
def test_graph_step_matches_the_step_loop(name, kwargs, num_envs, sequence, monkeypatch):
    calls = []
    short = stepping._host_single_step

    def counted(env, *args):
        out = short(env, *args)
        if env is fast:
            calls.append(out is not None)
        return out

    monkeypatch.setattr(stepping, "_host_single_step", counted)
    fast = make_or_skip(name, num_envs=num_envs, device="cuda", **kwargs)
    general = make_or_skip(name, num_envs=num_envs, device="cuda", **kwargs)
    with fast, general:
        general.__dict__["_host_step"] = None
        fast.reset(seed=5)
        general.reset(seed=5)
        rng = np.random.default_rng(2)
        previous = None
        for round_ in range(3):
            if round_ == 1:
                # A restored State and a new Variation change what the captured step reads.
                saved = general.get_state()
                fast.set_state(saved)
                general.set_state(saved)
                variation = general.sample_variation(rng=np.random.default_rng(round_))
                fast.set_variation(variation)
                general.set_variation(variation)
            for call in CALLS:
                action = fast.action_spec.sample(num_envs, rng=rng).cuda()
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


@pytest.mark.family("navigation")
def test_simple_point_maze_float64_actions_leave_the_graph():
    with make_or_skip("SimplePointMaze", num_envs=2, device="cuda") as fast:
        with make_or_skip("SimplePointMaze", num_envs=2, device="cuda") as general:
            general.__dict__["_host_step"] = None
            for env in (fast, general):
                env.reset(seed=3)
            rng = np.random.default_rng(4)
            for dtype in (torch.float32, torch.float64, torch.float32):
                action = fast.action_spec.sample(2, rng=rng).to("cuda", dtype)
                _same(fast.step(action, frames=None), general.step(action, frames=None))
            _same(fast.get_state(), general.get_state())
