"""Parity and buffer regressions specific to the TwoRoom JAX port."""

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state
from tests.helpers.tworoom import TorchTwoRoom

pytestmark = pytest.mark.family("navigation")


@pytest.mark.parametrize("seed", [0, 7, 31])
def test_public_steps_match_torch_dynamics(seed):
    with TorchTwoRoom(num_envs=8, device="cpu") as main, se.TwoRoom(num_envs=8, device="cpu") as env:
        main.reset(seed=seed)
        env.reset(seed=seed)
        assert env.execution_mode == "NATIVE_JAX"
        actions = torch.from_numpy(np.random.default_rng(seed).uniform(-2, 2, (40, 8, 2)).astype(np.float32))
        for action in actions:
            assert_same_state(main.step(action, frames=None), env.step(action, frames=None))


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
    with TorchTwoRoom(num_envs=7, device="cpu") as main, se.TwoRoom(num_envs=7, device="cpu") as env:
        for item in (main, env):
            item.set_variation(variation)
            item.reset(seed=0, options={"state": positions, "target_state": [180, 180]})
        for _ in range(6):
            assert_same_state(main.step(actions, frames=None), env.step(actions, frames=None))


def test_sequence_masks_freeze_each_lane_after_success():
    options = {"state": [[60, 112], [60, 112], [60, 112]], "target_state": [[60, 112], [85, 112], [180, 112]]}
    actions = torch.tensor([[[1.0, 0.0]] * 6] * 3)
    with TorchTwoRoom(num_envs=3, device="cpu") as main, se.TwoRoom(num_envs=3, device="cpu") as env:
        for item in (main, env):
            item.reset(seed=0, options=options)
        result = env.step(actions, frames=None)
        assert_same_state(main.step(actions, frames=None), result)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[65, 112], [70, 112], [90, 112.0]]))


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
    with TorchTwoRoom(num_envs=1, device="cpu") as main, se.TwoRoom(num_envs=1, device="cpu") as env:
        main.reset(seed=0)
        env.reset(seed=0)
        action = torch.arange(8, dtype=torch.float32).reshape(4, 2)[1:2]
        assert action.storage_offset() > 0
        assert_same_state(main.step(action, frames=None), env.step(action, frames=None))
        actions = (torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 32)[..., ::2]
        assert not actions.is_contiguous()
        assert_same_state(main.step(actions, frames=None), env.step(actions, frames=None))


def test_success_boundary_has_matching_boolean_outputs():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    target = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    with TorchTwoRoom(num_envs=128, device="cpu") as main, se.TwoRoom(num_envs=128, device="cpu") as env:
        for item in (main, env):
            item.reset(seed=0, options={"state": [60, 60], "target_state": target})
        assert_same_state(main.step(torch.zeros(128, 2), frames=None), env.step(torch.zeros(128, 2), frames=None))


def test_continuous_action_multiply_rounds_before_position_addition():
    options = {"state": [53.47578811645508, 60], "target_state": [180, 180]}
    action = torch.tensor([[-0.9868243932723999, 0.0]])
    with TorchTwoRoom(device="cpu") as main, se.TwoRoom(device="cpu") as env:
        for item in (main, env):
            item.reset(seed=0, options=options)
        expected, actual = main.step(action, frames=None), env.step(action, frames=None)
        assert actual["state", "agent", "position"][0, 0].item() == 48.541664123535156
        assert_same_state(expected, actual)
