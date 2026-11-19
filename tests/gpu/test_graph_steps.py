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
from tests.unit.test_host_single_step import _same, compare_runs  # noqa: E402

CLASSES = [
    *(pytest.param(name, {}, marks=pytest.mark.family("classic"), id=name)
      for name in ("CartPole", "Acrobot", "Pendulum", "MountainCar", "MountainCarContinuous")),
    *(pytest.param(name, {}, marks=pytest.mark.family("navigation"), id=name)
      for name in ("TwoRoomTarget", "PiecewiseZones", "Piecewise", "SimplePointMaze")),
    pytest.param("TwoRoom", {"simulation_backend": "torch"}, marks=pytest.mark.family("navigation"), id="TwoRoom"),
]  # fmt: skip
#: Each class on its CUDA graph step.
NAVIGATION = [
    ("TwoRoom", {"simulation_backend": "torch"}),
    ("TwoRoomTarget", {}),
    ("Piecewise", {}),
    ("PiecewiseZones", {}),
    ("SimplePointMaze", {}),
]


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
        compare_runs(fast, general, num_envs, sequence)
        graph = fast.__dict__.get("_graph") or fast._simulator._single
        # A Variation of the same shape and a restored State reuse the captured graph.
        assert len(graph._graphs) <= 2
    assert calls and all(calls)


@pytest.mark.family("navigation")
@pytest.mark.parametrize("name,kwargs", [("TwoRoomTarget", {}), ("TwoRoom", {"simulation_backend": "torch"})])
def test_a_new_variation_reuses_the_captured_graph(name, kwargs):
    with make_or_skip(name, num_envs=2, device="cuda", **kwargs) as env:
        env.reset(seed=0)
        action = torch.zeros(2, 2, device="cuda")
        env.step(action, frames=None)
        graph = env._graph
        for seed in range(10):
            env.set_variation(env.sample_variation(rng=np.random.default_rng(seed)))
            env.step(action, frames=None)
        assert len(graph._graphs) == 1


@pytest.mark.family("navigation")
def test_a_variation_set_under_inference_mode_steps_like_the_step_loop():
    with make_or_skip("TwoRoom", num_envs=2, device="cuda", simulation_backend="torch") as fast:
        with make_or_skip("TwoRoom", num_envs=2, device="cuda", simulation_backend="torch") as general:
            general.__dict__["_host_step"] = None
            variation = fast.sample_variation(rng=np.random.default_rng(1))
            action = torch.full((2, 2), 0.5, device="cuda")
            for env in (fast, general):
                env.reset(seed=0)
                env.step(action, frames=None)
                with torch.inference_mode():
                    env.set_variation(variation)
            for _ in range(3):
                _same(fast.step(action, frames=None), general.step(action, frames=None))


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


@pytest.mark.family("navigation")
@pytest.mark.parametrize("name,kwargs", NAVIGATION)
def test_a_cuda_step_to_a_non_finite_state_does_not_read_back(name, kwargs):
    """A CUDA step doesn't check finiteness, which would read back to the host; ``get_state`` still raises."""
    with make_or_skip(name, num_envs=2, device="cuda", **kwargs) as env:
        env.reset(seed=0)
        out = env.step(torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], device="cuda"), frames=None)
        assert env._graph._graphs, "the step did not run on the CUDA graph"
        assert any(bool(leaf.isnan().any()) for leaf in out["state"].values(True, True) if leaf.is_floating_point())
        with pytest.raises(ValueError, match="non-finite|nonfinite|NaN"):
            env.get_state()
