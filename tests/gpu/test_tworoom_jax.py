"""CUDA parity, device residency and stream ownership for TwoRoom's DLPack backend."""

from contextlib import closing

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.semantics import assert_same_state, compare_state

pytestmark = [
    pytest.mark.family("navigation"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("batch", [1, 64, 4096])
@pytest.mark.swm
def test_cuda_batched_steps_match_cpu_and_swm_with_arbitrary_actions(batch):
    # Sample source lanes at large batches: each source env owns full-resolution render grids.
    lanes = np.linspace(0, batch - 1, min(batch, 64), dtype=int).tolist()
    with (
        closing(NativeNavigationReference("two_room", len(lanes))) as source,
        se.TwoRoom(num_envs=batch, device="cpu") as cpu,
        se.TwoRoom(num_envs=batch, device="cuda") as env,
    ):
        env.reset(seed=7)
        compare_state(env.get_state(), cpu.reset(seed=7))
        for lane, source_env in zip(lanes, source.envs, strict=True):
            source_env.reset(seed=7 + lane)
        compare_state(env.get_state()[lanes], source.get_state())
        generator = torch.Generator(device="cuda").manual_seed(12)
        for _ in range(32):
            action = (torch.rand(batch, 4, generator=generator, device="cuda") * 4 - 2)[:, ::2]
            assert not action.is_contiguous()
            actual = env.step(action, frames=None, states=None).exclude("valid")
            compare_state(actual, cpu.step(action.cpu(), frames=None, states=None).exclude("valid"))
            compare_state(actual[lanes], source.step(action[lanes]))


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
    with se.TwoRoom(num_envs=3, device="cuda") as env:
        env.set_variation({"wall.axis": [0], "door.number": [3], "agent.speed": [5.25]})
        env.reset(seed=0, options={"state": [60, 60], "target_state": [[60, 60], [90, 60], [180, 180]]})
        actions = torch.zeros(3, 8, 2, device="cuda")
        actions[..., 0] = 1
        result = env.step(actions, frames=None)
        expected = torch.tensor([[65.25, 60], [75.75, 60], [102, 60]], device="cuda")
        assert_same_state(result["state", "agent", "position"], expected)
        assert result["valid"].sum(dim=1).tolist() == [1, 3, 8]


@pytest.mark.swm
def test_cuda_success_boundary_matches_swm():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    targets = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    with (
        closing(NativeNavigationReference("two_room", len(targets))) as source,
        se.TwoRoom(num_envs=128, device="cuda") as env,
    ):
        state = env.reset(seed=0, options={"state": [60, 60], "target_state": targets})
        source.reset(seed=0)
        source.set_state(state)
        action = torch.zeros(128, 2, device="cuda")
        expected = source.step(action)
        actual = env.step(action, states=None, frames=None).exclude("valid")
        compare_state(actual, expected)


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise"])
def test_cuda_shared_distance_matches_cpu_norm_without_host_conversion(name, monkeypatch):
    targets = np.random.default_rng(7).uniform(14, 209, (512, 2)).astype(np.float32)
    with se.make(name, num_envs=len(targets), device="cuda") as env:
        state = env.reset(seed=0, options={"state": [60, 60], "target_state": targets})
        delta = (state["agent", "position"] - state["target", "position"]).cpu()
        expected = torch.tensor([float(torch.norm(row)) for row in delta], dtype=torch.float64)

        def forbidden(*args, **kwargs):
            raise AssertionError("host conversion in CUDA shared navigation outputs")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            result = env._outputs(state)
        assert_same_state(result["info", "distance_to_target"].cpu(), expected)
        assert_same_state(result["terminated"].cpu(), expected < 16)


def test_cuda_single_lane_sequence_with_offset_action_views():
    with se.TwoRoom(num_envs=1, device="cuda") as env:
        env.reset(seed=0, options={"state": [50, 50], "target_state": [180, 180]})
        actions = torch.arange(12, device="cuda", dtype=torch.float32).reshape(1, 6, 2) / 32
        actual = env.step(actions, frames=None)
        expected = torch.tensor([[54.6875, 55.625]], device="cuda")
        assert_same_state(actual["state", "agent", "position"], expected)
