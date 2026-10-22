"""CUDA parity, device residency and stream ownership for TwoRoom's DLPack backend."""

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state
from tests.helpers.tworoom import TorchTwoRoom

pytestmark = [
    pytest.mark.family("navigation"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("batch", [1, 64, 4096])
def test_cuda_public_step_matches_torch_with_arbitrary_actions(batch):
    with (
        TorchTwoRoom(num_envs=batch, device="cuda") as main,
        se.TwoRoom(num_envs=batch, device="cuda") as env,
    ):
        main.reset(seed=7)
        env.reset(seed=7)
        generator = torch.Generator(device="cuda").manual_seed(12)
        for _ in range(32):
            action = (torch.rand(batch, 4, generator=generator, device="cuda") * 4 - 2)[:, ::2]
            assert not action.is_contiguous()
            expected = main.step(action, frames=None, states=None)
            actual = env.step(action, frames=None, states=None)
            assert_same_state(actual, expected)


def test_cuda_dlpack_step_on_nondefault_stream_has_no_host_conversion(monkeypatch):
    from stable_envs.envs.navigation.tworoom_jax import from_torch

    def forbidden(*args, **kwargs):
        raise AssertionError("host conversion in CUDA JAX TwoRoom step")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), se.TwoRoom(num_envs=32, device="cuda") as env:
        env.reset(seed=0, options={"state": [50, 50], "target_state": [180, 180]})
        action = torch.zeros(32, 2, device="cuda")
        borrowed = from_torch(action)
        assert borrowed.unsafe_buffer_pointer() == action.data_ptr()
        assert torch.from_dlpack(borrowed).data_ptr() == action.data_ptr()
        del borrowed
        env.step(action, states=None, frames=None)
        initial = env.get_state()
        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            action.fill_(0.125)
            result = env.step(action, states=None, frames=None)
            held = result.clone()
            action.fill_(-0.25)
            env.step(action, states=None, frames=None)
        stream.synchronize()
        assert_same_state(result, held)
        assert_same_state(result["state", "agent", "position"], initial["agent", "position"] + 0.625)


def test_cuda_masked_sequence_with_variation():
    with TorchTwoRoom(num_envs=3, device="cuda") as main, se.TwoRoom(num_envs=3, device="cuda") as env:
        for item in (main, env):
            item.set_variation({"wall.axis": [0], "door.number": [3], "agent.speed": [5.25]})
            item.reset(seed=0, options={"state": [60, 60], "target_state": [[60, 60], [90, 60], [180, 180]]})
        actions = torch.ones(3, 8, 2, device="cuda")
        assert_same_state(main.step(actions, frames=None), env.step(actions, frames=None))


def test_cuda_success_boundary_matches_main_cpu_norm():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    targets = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    with (
        TorchTwoRoom(num_envs=128, device="cuda") as main,
        se.TwoRoom(num_envs=128, device="cuda") as env,
    ):
        for item in (main, env):
            item.reset(seed=0, options={"state": [60, 60], "target_state": targets})
        action = torch.zeros(128, 2, device="cuda")
        expected, actual = main.step(action, frames=None), env.step(action, frames=None)
        assert_same_state(actual, expected)


def test_cuda_single_lane_sequence_with_offset_action_views():
    with TorchTwoRoom(num_envs=1, device="cuda") as main, se.TwoRoom(num_envs=1, device="cuda") as env:
        for item in (main, env):
            item.reset(seed=0, options={"state": [50, 50], "target_state": [180, 180]})
        actions = torch.arange(12, device="cuda", dtype=torch.float32).reshape(1, 6, 2) / 32
        expected, actual = main.step(actions, frames=None), env.step(actions, frames=None)
        assert_same_state(actual, expected)
