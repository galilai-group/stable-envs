"""Caller-stream changes must order public checkpoints as well as Warp physics."""

import pytest
import torch

from tests.helpers.reference.dmc_mjwarp import require_cuda
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.family("dmc")


@pytest.fixture(scope="module", autouse=True)
def cuda_required():
    require_cuda()


def test_reset_step_and_snapshot_across_caller_streams():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    device = torch.device("cuda", torch.cuda.current_device())
    first, second = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(first):
        adapter = DMCWarpAdapter("reacher", 3, "cuda", 31)
        initial = adapter.get_state()
    try:
        # No caller synchronization is inserted between adapter operations.
        with torch.cuda.stream(second):
            initial_on_second = adapter.get_state()
            first_step = adapter.step(torch.full((3, 2), 0.25, device=device, dtype=torch.float64))
        with torch.cuda.stream(first):
            first_step_on_first = adapter.get_state()
            second_step = adapter.step(torch.full((3, 2), -0.25, device=device, dtype=torch.float64))
        with torch.cuda.stream(second):
            second_step_on_second = adapter.get_state()
            reset = adapter.reset(53)
        with torch.cuda.stream(first):
            reset_on_first = adapter.get_state()
            adapter.step(torch.zeros((3, 2), device=device, dtype=torch.float64), capture_state=False)
        first.synchronize()
        second.synchronize()

        assert_same_state(initial_on_second, initial)
        assert_same_state(first_step_on_first, first_step["state"])
        assert_same_state(second_step_on_second, second_step["state"])
        assert_same_state(reset_on_first, reset)
        assert initial["lifecycle", "step_count"].flatten().tolist() == [0, 0, 0]
        assert first_step["state", "lifecycle", "step_count"].flatten().tolist() == [2, 2, 2]
        assert second_step["state", "lifecycle", "step_count"].flatten().tolist() == [4, 4, 4]
        assert all(value.device == device for value in second_step.values(True, True))

        # On a multi-GPU host, an unindexed construction device must stay bound
        # to its original GPU after the caller changes the current CUDA device.
        # A single-GPU acceptance run still executes all required assertions.
        other = (device.index + 1) % torch.cuda.device_count()
        with torch.cuda.device(other):
            output = adapter.step(torch.zeros((3, 2), device=device, dtype=torch.float64))
            snapshot = adapter.get_state()
        torch.cuda.synchronize(device)
        assert adapter.device == device
        assert all(value.device == device for value in output.values(True, True))
        assert_same_state(snapshot, output["state"])
    finally:
        adapter.close()


def test_prepared_restore_and_owned_state_across_caller_streams():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    device = torch.device("cuda", torch.cuda.current_device())
    first, second = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(first):
        adapter = DMCWarpAdapter("reacher", 3, device, 71)
        checkpoint = adapter.get_state()
        expected = checkpoint.clone()
    try:
        with torch.cuda.stream(second):
            live = adapter.reset(83)
        with torch.cuda.stream(first):
            adapter.prepare_restore(checkpoint)
            # Mutating the supplied object after prepare cannot alter the
            # adapter-owned staged checkpoint, even before the other stream runs.
            checkpoint["qpos"].add_(1)
        with torch.cuda.stream(second):
            before_commit = adapter.get_state()
            adapter.commit_restore()
            committed = adapter.get_state()
        with torch.cuda.stream(first):
            committed_on_first = adapter.get_state()
            adapter.step(torch.zeros((3, 2), device=device, dtype=torch.float64))
        first.synchronize()
        second.synchronize()

        assert_same_state(before_commit, live)
        assert_same_state(committed, expected)
        assert_same_state(committed_on_first, expected)
        # Discard has no GPU work and must leave both the live and staged state
        # lifecycle independent of subsequent public calls on the other stream.
        with torch.cuda.stream(second):
            before_discard = adapter.get_state()
            adapter.prepare_restore(expected)
            adapter.discard_restore()
        with torch.cuda.stream(first):
            after_discard = adapter.get_state()
            with pytest.raises(RuntimeError, match="No restore is prepared"):
                adapter.commit_restore()
        first.synchronize()
        second.synchronize()
        assert_same_state(after_discard, before_discard)
    finally:
        adapter.close()
