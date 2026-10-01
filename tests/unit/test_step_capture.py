"""Public stepping: shape dispatch, capture, early stops and snapshot ownership."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv
from stable_envs.spaces import Dict


class Counter(snv.Env):
    action_dim = 1
    action_dtype = torch.int64
    state_dim = 1

    def __init__(self, device="cpu"):
        super().__init__(num_envs=3, device=device)
        self.position = torch.zeros(3, 1, dtype=torch.int64, device=self.device)
        self.limit = torch.tensor([1, 3, 20], device=self.device)
        self.calls = torch.zeros(3, dtype=torch.int64, device=self.device)
        self.render_calls = 0

    def make_variation_space(self):
        return Dict({})

    def get_state(self):
        return TensorDict({"position": self.position.clone()}, [3], device=self.device)

    def _step(self, actions, *, active=None):
        active = torch.ones(3, dtype=torch.bool, device=self.device) if active is None else active
        self.position[active] += torch.as_tensor(actions, device=self.device)[active]
        self.calls += active
        return self._step_result(
            self.get_state(), reward=self.position[:, 0].float(), terminated=self.position[:, 0] >= self.limit
        )

    def render(self, state=None):
        self.render_calls += 1
        state = self.get_state() if state is None else state
        return state["position"].to(torch.uint8).reshape(3, 1, 1, 1).expand(3, 2, 2, 3).clone()


@pytest.mark.parametrize("capture", [None, "states", "final_frame", "all_frames"])
@pytest.mark.parametrize("sequence", [False, True])
def test_capture_contract(capture, sequence):
    with Counter() as env:
        actions = torch.ones((3, 4, 1) if sequence else (3, 1), dtype=torch.int64)
        out = env.step(actions, capture=capture)
        assert out.batch_size == torch.Size([3])
        assert ("states" in out.keys()) == (sequence and capture in (None, "states"))
        assert ("frame" in out.keys()) == (capture != "states")
        assert ("frames" in out.keys()) == (capture == "all_frames")
        assert env.render_calls == (0 if capture == "states" else 4 if sequence and capture == "all_frames" else 1)
        assert torch.equal(out["state"]["position"], env.position)
        if sequence:
            assert out["valid"].tolist() == [[True, False, False, False], [True, True, True, False], [True] * 4]
            assert env.calls.tolist() == [1, 3, 4]
            assert out["reward"].tolist() == [[1, 0, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]]
            if "states" in out.keys():
                assert out["states"]["position"][:, :, 0].tolist() == [[1] * 4, [1, 2, 3, 3], [1, 2, 3, 4]]
        held = out.clone()
        env.step(torch.ones(3, 1, dtype=torch.int64), capture="states")
        assert (out == held).all()


@pytest.mark.parametrize("shape", [(3, 0, 1), (3, 2), (2, 3, 1), (3, 1, 1, 1)])
def test_invalid_shape_refused_before_mutation(shape):
    with Counter() as env:
        with pytest.raises(ValueError, match="actions must"):
            env.step(torch.zeros(shape), capture="states")
        assert not env.calls.any()


def test_invalid_capture_and_symbolic_refused_before_mutation():
    with Counter() as env:
        with pytest.raises(ValueError, match="capture must"):
            env.step(torch.ones(3, 1), capture="invalid")
        env.produces_rgb = False
        with pytest.raises(ValueError, match="no RGB"):
            env.step(torch.ones(3, 1))
        assert not env.calls.any()
        env.step(torch.ones(3, 1, dtype=torch.int64), capture="states")


def test_one_step_sequence_keeps_time_axis():
    with Counter() as env:
        out = env.step(torch.ones(3, 1, 1, dtype=torch.int64), capture="states")
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
        expected = [env.step(actions[:, t], capture="states").clone() for t in range(4)]
        env.set_state(start)
        out = env.step(actions, capture="states")
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
        out = env.step(torch.zeros(3, 5, 1, dtype=torch.int64), capture="states")
        assert calls == [1, 2, 3]
        assert out["valid"].sum(1).tolist() == calls
        assert out["state"]["lifecycle", "elapsed_steps"][:, 0].tolist() == calls


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
        expected = [env.step(actions[:, t], capture="states")["state"] for t in range(3)]
        env.set_state(state)
        out = env.step(actions, capture="states")
        for t in range(3):
            assert (out["states"][:, t] == expected[t]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation runs on Oscar")
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
        result = env.step(actions, capture="all_frames")
        assert result["frame"].device.type == "cuda"
        assert torch.equal(result["frames"], torch.stack([out["frame"] for out in reference], dim=1))
        assert (result["state"] == reference[-1]["state"]).all()
        env.set_state(initial)
        final = env.step(actions, capture="final_frame")
        assert (final["state"] == result["state"]).all()
        assert torch.equal(final["frame"], result["frame"])
