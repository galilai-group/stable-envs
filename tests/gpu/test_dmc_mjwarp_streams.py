"""Caller-stream changes must order public checkpoints as well as Warp physics."""

import pytest
import torch

from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_reset_step_and_restore_across_caller_streams():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    device = torch.device("cuda", torch.cuda.current_device())
    first, second = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(first):
        adapter = DMCWarpAdapter("cartpole-balance", 3, device, 31)
        initial = adapter.get_state()
    try:
        # No caller synchronization occurs between adapter operations.
        with torch.cuda.stream(second):
            initial_on_second = adapter.get_state()
            stepped = adapter.step(torch.full((3, 1), 0.25, device=device, dtype=torch.float64))
        with torch.cuda.stream(first):
            stepped_on_first = adapter.get_state()
            reset = adapter.reset(53)
        with torch.cuda.stream(second):
            reset_on_second = adapter.get_state()
            adapter.prepare_restore(initial)
        with torch.cuda.stream(first):
            adapter.commit_restore()
            restored = adapter.get_state()
        first.synchronize()
        second.synchronize()
        assert_same_state(initial_on_second, initial)
        assert_same_state(stepped_on_first, stepped["state"])
        assert_same_state(reset_on_second, reset)
        assert_same_state(restored, initial)
        assert stepped["state", "lifecycle", "step_count"].flatten().tolist() == [2, 2, 2]
        assert all(value.device == device for value in stepped.values(True, True))
    finally:
        adapter.close()
