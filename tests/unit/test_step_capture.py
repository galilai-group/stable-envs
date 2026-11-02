"""Public stepping: shape dispatch, capture, early stops and snapshot ownership."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv
from tests.helpers.counter_env import Counter, PipelinedCounter


@pytest.mark.parametrize("env_type", [Counter, PipelinedCounter])
@pytest.mark.parametrize("states", [None, slice(None), slice(-1, None), slice(None, None, -1), slice(9, 12)])
@pytest.mark.parametrize(
    "frames", [None, slice(None), slice(-1, None), slice(None, None, 2), slice(None, None, -1), slice(9, 12)]
)
@pytest.mark.parametrize("sequence", [False, True])
def test_slice_contract(states, frames, sequence, env_type):
    with env_type() as env:
        length = 4 if sequence else 1
        actions = torch.ones((3, length, 1) if sequence else (3, 1), dtype=torch.int64)
        out = env.step(actions, states=states, frames=frames)
        positions = torch.tensor([[1, 1, 1, 1], [1, 2, 3, 3], [1, 2, 3, 4]])[:, :length]
        valid = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)[:, :length]
        assert out.batch_size == torch.Size([3])
        assert ("states" in out.keys()) == (states is not None)
        assert ("frames" in out.keys()) == (frames is not None)
        assert "frame" not in out.keys()
        assert env.render_calls == (0 if frames is None else len(range(length)[frames]))
        assert torch.equal(out["state"]["position"], positions[:, -1:])
        assert torch.equal(out["valid"], valid)
        assert env.calls.tolist() == ([1, 3, 4] if sequence else [1, 1, 1])
        if states is not None:
            indices = list(range(length)[states])
            assert out["states"].batch_size == torch.Size([3, len(indices)])
            assert torch.equal(out["states"]["position"][:, :, 0], positions[:, indices])
            assert torch.equal(out["state_valid"], valid[:, indices])
        if frames is not None:
            indices = list(range(length)[frames])
            assert out["frames"].shape == (3, len(indices), 2, 2, 3)
            assert torch.equal(out["frames"][:, :, 0, 0, 0], positions[:, indices].to(torch.uint8))
            assert torch.equal(out["frame_valid"], valid[:, indices])
        held = out.clone()
        env.step(torch.ones(3, 1, dtype=torch.int64), frames=None)
        assert (out == held).all()


def test_pipeline_overlaps_only_known_future_actions_and_returns_idle():
    with PipelinedCounter() as env:
        env.step(torch.ones(3, 8, 1, dtype=torch.int64), frames=slice(None))
        assert env.overlaps == 7
        assert env.pending is None
        env.overlaps = 0
        env.step(torch.ones(3, 1, dtype=torch.int64))
        assert env.overlaps == 0
        assert env.pending is None


def test_pipeline_renderer_exception_aborts_inflight_transition(monkeypatch):
    with PipelinedCounter() as env:

        def fail(state):
            assert env.pending is not None
            raise RuntimeError("renderer failed")

        monkeypatch.setattr(env, "render", fail)
        with pytest.raises(RuntimeError, match="renderer failed"):
            env.step(torch.ones(3, 8, 1, dtype=torch.int64), frames=slice(None))
        assert env.aborted and env.pending is None


def test_pipeline_all_lanes_finish_before_first_selected_snapshot():
    with Counter() as reference, PipelinedCounter() as candidate:
        reference.limit.fill_(1)
        candidate.limit.fill_(1)
        actions = torch.ones(3, 8, 1, dtype=torch.int64)
        expected = reference.step(actions, states=None, frames=slice(-1, None))
        actual = candidate.step(actions, states=None, frames=slice(-1, None))
        assert (expected == actual).all()
        assert candidate.calls.tolist() == [1, 1, 1]
        assert candidate.pending is None


@pytest.mark.parametrize("shape", [(3, 0, 1), (3, 2), (2, 3, 1), (3, 1, 1, 1)])
def test_invalid_shape_refused_before_mutation(shape):
    with Counter() as env:
        with pytest.raises(ValueError, match="actions must"):
            env.step(torch.zeros(shape), frames=None)
        assert not env.calls.any()


@pytest.mark.parametrize("name", ["states", "frames"])
@pytest.mark.parametrize("selector", [-1, "all", [0, 1], slice(None, None, 0), slice(1.5, None)])
def test_invalid_selectors_before_mutation(name, selector):
    with Counter() as env:
        with pytest.raises((TypeError, ValueError)):
            env.step(torch.ones(3, 1), **{name: selector})
        assert not env.calls.any()


def test_symbolic_requires_no_frame_selection_before_mutation():
    with Counter() as env:
        env.produces_rgb = False
        with pytest.raises(ValueError, match="no RGB"):
            env.step(torch.ones(3, 1))
        assert not env.calls.any()
        with pytest.raises(ValueError, match="no RGB"):
            env.step(torch.ones(3, 1, dtype=torch.int64), frames=slice(1, 1))
        assert not env.calls.any()
        env.step(torch.ones(3, 1, dtype=torch.int64), frames=None)
        assert env.render_calls == 0


def test_final_only_exports_one_snapshot_and_sparse_export_skips_unselected():
    class CheapAdvance(Counter):
        exports = 0

        def get_state(self):
            self.exports += 1
            return super().get_state()

        def _advance_without_snapshot(self, actions, *, active=None):
            active = torch.ones(3, dtype=torch.bool) if active is None else active
            self.position[active] += torch.as_tensor(actions)[active]
            self.calls += active
            return TensorDict({"terminated": self.position[:, 0] >= self.limit}, [3])

    with CheapAdvance() as env:
        env.step(torch.ones(3, 4, 1, dtype=torch.int64), states=None)
        assert env.exports == 1
    with CheapAdvance() as env:
        env.step(torch.ones(3, 4, 1, dtype=torch.int64), states=slice(None, None, 2), frames=None)
        assert env.exports == 3  # selected 0,2, plus mandatory final State


@pytest.mark.family("dmc")
@pytest.mark.parametrize("backend", ["native", "mjx", "mjwarp"])
def test_dmc_snapshot_capture_forwarding(backend):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from stable_envs.envs.dmcontrol._delegated import DelegatedSemantics

    env = SimpleNamespace(simulation_backend=backend, _check_open=Mock(), _simulator=Mock(), _step=Mock())
    actions = torch.zeros(3, 1)
    result = DelegatedSemantics._advance_without_snapshot(env, actions)
    if backend == "mjwarp":
        env._simulator.step.assert_called_once_with(actions, capture_state=False)
        assert result is env._simulator.step.return_value
        env._step.assert_not_called()
    else:
        env._step.assert_called_once_with(actions)
        assert result is env._step.return_value
        env._simulator.step.assert_not_called()


def test_one_step_sequence_keeps_time_axis():
    with Counter() as env:
        out = env.step(torch.ones(3, 1, 1, dtype=torch.int64), frames=None)
        assert out["states"].batch_size == torch.Size([3, 1])
        assert out["reward"].shape == (3, 1)


def test_public_functional_api_removed():
    with Counter() as env:
        assert not hasattr(env, "unroll")
        assert not hasattr(env, "transition")
        assert not hasattr(env, "step_sequence")
    assert not hasattr(snv, "Dynamics")


@pytest.mark.parametrize("name,dtype", [("CartPole", np.int64), ("Pendulum", np.float64)])
def test_accepted_classic_matches_repeated_steps(name, dtype):
    with snv.make(name, num_envs=3, device="cpu") as env:
        start = env.reset(seed=17)
        actions = np.zeros((3, 4, 1), dtype=dtype)
        expected = [env.step(actions[:, t], frames=None).clone() for t in range(4)]
        env.set_state(start)
        out = env.step(actions, frames=None)
        for t, reference in enumerate(expected):
            assert (out["states"][:, t] == reference["state"]).all()
            assert torch.equal(out["reward"][:, t], reference["reward"])
        assert (env.get_state() == expected[-1]["state"]).all()


def test_classic_stops_at_distinct_time_limits_without_step_or_reset(monkeypatch):
    with snv.CartPole(num_envs=3, device="cpu") as env:
        env.reset(seed=17)
        for lane, upstream in enumerate(env._simulator.envs):
            upstream._max_episode_steps = lane + 1
        calls = [0, 0, 0]
        for lane, upstream in enumerate(env._simulator.envs):
            original = upstream.step

            def counted(action, lane=lane, original=original):
                calls[lane] += 1
                return original(action)

            monkeypatch.setattr(upstream, "step", counted)
        out = env.step(torch.zeros(3, 5, 1, dtype=torch.int64), frames=None)
        assert calls == [1, 2, 3]
        assert out["valid"].sum(1).tolist() == calls
        assert out["state"]["lifecycle", "elapsed_steps"][:, 0].tolist() == calls


@pytest.mark.family("navigation")
@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise", "SimplePointMaze"])
def test_navigation_sequence_matches_scalar_calls(name):
    if name == "SimplePointMaze":
        pytest.importorskip("matplotlib")
    with snv.make(name, num_envs=3, device="cpu") as env:
        state = env.reset(seed=17)
        if name in ("TwoRoom", "Piecewise"):
            state["target", "position"].fill_(0)
            env.set_state(state)
        actions = torch.zeros(3, 3, env.action_dim)
        expected = [env.step(actions[:, t], frames=None)["state"] for t in range(3)]
        env.set_state(state)
        out = env.step(actions, frames=None)
        for t in range(3):
            assert (out["states"][:, t] == expected[t]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", ["CartPole", "Pendulum", "TwoRoom", "Piecewise"])
def test_gpu_sequence_frames_equal_repeated_steps(name):
    with snv.make(name, num_envs=3, device="cuda") as env:
        initial = env.reset(seed=17)
        if name in ("TwoRoom", "Piecewise"):
            initial["target", "position"].fill_(0)
            env.set_state(initial)
        actions = torch.zeros(3, 3, env.action_dim, dtype=env.action_spec.dtype, device="cuda")
        reference = [env.step(actions[:, t]) for t in range(3)]
        env.set_state(initial)
        result = env.step(actions, states=None, frames=slice(None))
        assert result["frames"].device.type == "cuda"
        assert torch.equal(result["frames"], torch.stack([out["frames"][:, 0] for out in reference], dim=1))
        assert (result["state"] == reference[-1]["state"]).all()
        env.set_state(initial)
        final = env.step(actions, states=None)
        assert (final["state"] == result["state"]).all()
        assert torch.equal(final["frames"][:, 0], result["frames"][:, -1])

        for selection in (slice(None, None, 2), slice(None, None, -1), slice(2, 2)):
            env.set_state(initial)
            selected = env.step(actions, states=selection, frames=selection)
            indices = list(range(3)[selection])
            assert torch.equal(selected["frames"], result["frames"][:, indices])
            assert selected["states"].batch_size == torch.Size([3, len(indices)])
            for j, t in enumerate(indices):
                assert (selected["states"][:, j] == reference[t]["state"]).all()
