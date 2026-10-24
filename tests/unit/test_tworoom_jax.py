"""Parity and buffer regressions specific to the TwoRoom JAX port."""

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.family("navigation")


def test_collisions_with_fractional_per_lane_geometry():
    positions = [[95, 100], [130, 100], [100, 49], [50, 21], [202, 100], [49, 100], [100, 130]]
    actions = torch.tensor([[1, 0], [-1, 0], [1, 0], [0, -5], [5, 0], [0, 1], [0, -1]], dtype=torch.float64)
    variation = {
        "agent.radius": [[7.125], [7.5], [8.25], [7], [7], [7], [8.125]],
        "agent.speed": [[5.5], [6.25], [4.125], [5], [5], [5], [7.25]],
        "wall.axis": [[1], [1], [1], [1], [1], [0], [0]],
        "wall.thickness": [[11.75], [10.25], [9.75], [10], [10], [13.5], [15.75]],
        "door.number": [[0.9], [1], [2.75], [1], [1], [3], [2]],
        "door.position": [[49, 112, 180]] * 7,
        "door.size": [[14.25, 10.5, 7.75]] * 7,
    }
    with se.TwoRoom(num_envs=7, device="cpu") as env:
        env.set_variation(variation)
        env.reset(seed=0, options={"state": positions, "target_state": [180, 180]})
        for step in range(1, 7):
            result = env.step(actions, frames=None)
            expected = torch.tensor(
                [
                    [99.375, 100],
                    [125, 100],
                    [100 + 4.125 * step, 49],
                    [50, 21],
                    [203, 100],
                    [49, 100 + 5 * step],
                    [100, 130 - 7.25 * step],
                ]
            )
            assert_same_state(result["state", "agent", "position"], expected)


def test_sequence_masks_freeze_each_lane_after_success():
    options = {"state": [[60, 112], [60, 112], [60, 112]], "target_state": [[60, 112], [85, 112], [180, 112]]}
    actions = torch.tensor([[[1.0, 0.0]] * 6] * 3)
    with se.TwoRoom(num_envs=3, device="cpu") as env:
        env.reset(seed=0, options=options)
        result = env.step(actions, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[65, 112], [70, 112], [90, 112.0]]))
        assert result["valid"].sum(dim=1).tolist() == [1, 2, 6]


def test_dlpack_shares_contiguous_inputs_and_step_has_no_host_conversion(monkeypatch):
    from stable_envs.envs.navigation.tworoom_jax import from_torch

    value = torch.ones(4, 2)
    borrowed = from_torch(value)
    assert borrowed.unsafe_buffer_pointer() == value.data_ptr()
    assert torch.from_dlpack(borrowed).data_ptr() == value.data_ptr()
    with se.TwoRoom(num_envs=4, device="cpu") as env:
        env.reset(seed=0)
        action = torch.ones(4, 2, requires_grad=True)
        env.step(action, states=None, frames=None)  # Compile before guarding the runtime path.
        x64 = jax.config.x64_enabled

        def forbidden(*args, **kwargs):
            raise AssertionError("host conversion in JAX TwoRoom step")

        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            result = env.step(action, states=None, frames=None)
            held = result.clone()
            env.step(-action, states=None, frames=None)
        assert_same_state(result, held)
        assert jax.config.x64_enabled == x64


def test_noncontiguous_actions_and_one_lane_offset_views():
    with se.TwoRoom(num_envs=1, device="cpu") as env:
        env.reset(seed=0, options={"state": [50, 50], "target_state": [180, 180]})
        action = torch.arange(8, dtype=torch.float32).reshape(4, 2)[1:2]
        assert action.storage_offset() > 0
        result = env.step(action, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[55.0, 55.0]]))
        actions = (torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 32)[..., ::2]
        assert not actions.is_contiguous()
        result = env.step(actions, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[61.25, 62.8125]]))


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise"])
def test_shared_distance_matches_cpu_norm_without_host_conversion(name, monkeypatch):
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    target = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    target = np.concatenate((target, np.random.default_rng(7).uniform(14, 209, (512, 2)).astype(np.float32)))
    with se.make(name, num_envs=len(target), device="cpu") as env:
        state = env.reset(seed=0, options={"state": [60, 60], "target_state": target})
        delta = state["agent", "position"] - state["target", "position"]
        expected = torch.tensor([float(torch.norm(row)) for row in delta], dtype=torch.float64)

        def forbidden(*args, **kwargs):
            raise AssertionError("host conversion in shared navigation outputs")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            result = env._outputs(state)
        assert_same_state(result["info", "distance_to_target"], expected)
        assert_same_state(result["terminated"], expected < 16)


def test_step_uses_the_observation_hook(monkeypatch):
    with se.TwoRoom(device="cpu") as env:
        env.reset(seed=0)
        expected = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        monkeypatch.setattr(env, "_observation", lambda agent, target: expected.clone())
        assert_same_state(env.step(torch.zeros(1, 2), frames=None)["observation"], expected)


def test_continuous_action_multiply_rounds_before_position_addition():
    options = {"state": [53.47578811645508, 60], "target_state": [180, 180]}
    action = torch.tensor([[-0.9868243932723999, 0.0]])
    with se.TwoRoom(device="cpu") as env:
        env.reset(seed=0, options=options)
        actual = env.step(action, frames=None)
        assert actual["state", "agent", "position"][0, 0].item() == 48.541664123535156
