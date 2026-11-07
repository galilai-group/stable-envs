"""Persistent Maze ownership, atomic validation, RNG isolation and lookahead."""

import sys
from functools import partial

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as se
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter, _resting_lanes
from stable_envs.envs.ogbench.simulation.maze_process import OGBMazeProcessAdapter
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize(
    "dtype,body", [(np.float32, {}), (np.float64, {}), (np.float32, dict(loco="ant", maze="teleport"))]
)
def test_complete_state_rng_outputs_and_inactive_lanes(dtype, body):
    local, remote = OGBMazeAdapter(5, seed=17, **body), OGBMazeProcessAdapter(5, seed=17, processes=2, **body)
    try:
        assert_same_state(local.get_state(), remote.get_state())
        initial = local.get_state()
        initial["rng", "global_has_gauss"][4] = True
        initial["rng", "global_cached_gaussian"][4] = 0.125
        initial["rng", "env_has_uint32"][4] = 1
        initial["rng", "env_uinteger"][4] = 2**32 - 1
        for adapter in (local, remote):
            adapter.set_state(initial)
        actions = np.random.default_rng(31).uniform(-0.5, 0.5, (5, 8, local.action_dim)).astype(dtype)
        held = remote.get_state()
        for t in range(8):
            mask = None if t == 0 else torch.tensor([True, False, True, False, True])
            assert_same_state(local.step(actions[:, t], active=mask), remote.step(actions[:, t], active=mask))
        assert_same_state(
            local.step(actions[:, 0], capture_state=False), remote.step(actions[:, 0], capture_state=False)
        )
        assert_same_state(local.get_state(), remote.get_state())
        assert_same_state(held, initial)
        for adapter in (local, remote):
            adapter.set_state(initial)
        assert_same_state(local.reset(), remote.reset())
        assert_same_state(local.reset(42), remote.reset(42))
        edited = remote.get_state()
        edited["root", "position"].zero_()
        assert_same_state(local.get_state(), remote.get_state())
    finally:
        local.close()
        remote.close()
    assert all(not p.is_alive() for p in remote._processes)


def test_rejected_restore_keeps_every_shard_usable():
    adapter = OGBMazeProcessAdapter(5, seed=17, processes=2)
    try:
        initial = adapter.get_state()
        invalid = initial.clone()
        invalid["root", "position"][0] += 0.25  # First shard must not commit this.
        invalid["task", "id"][4] = 999  # Last shard rejects preparation.
        with pytest.raises(ValueError, match="task id"):
            adapter.set_state(invalid)
        assert not adapter._closed and adapter._pending is None
        assert_same_state(initial, adapter.get_state())
        for call in (
            lambda: adapter.reset(-1),
            lambda: adapter.step(np.zeros((4, 2))),
            lambda: adapter.step(np.zeros((5, 2)), active=np.ones(5)),
        ):
            with pytest.raises(ValueError):
                call()
            assert_same_state(initial, adapter.get_state())
        adapter.step(np.zeros((5, 2)))
        adapter.set_state(initial)
        assert_same_state(initial, adapter.get_state())
    finally:
        adapter.close()


def test_pending_request_exclusion_owned_output_and_close():
    adapter = OGBMazeProcessAdapter(3, seed=17, processes=2)
    try:
        initial = adapter.get_state()
        adapter.submit_step(np.zeros((3, 2)), capture_state=False)
        for call in (
            adapter.get_state,
            lambda: adapter.set_state(initial),
            adapter.reset,
            lambda: adapter.submit_step(np.zeros((3, 2))),
        ):
            with pytest.raises(RuntimeError, match="in flight"):
                call()
        result = adapter.receive_step()
        assert "state" not in result.keys()
        held = result.clone()
        adapter.step(np.ones((3, 2)))
        assert_same_state(result, held)
        adapter.submit_step(np.zeros((3, 2)))
    finally:
        adapter.close()
    assert all(not p.is_alive() for p in adapter._processes)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()


def test_worker_death_fails_closed():
    adapter = OGBMazeProcessAdapter(3, processes=2)
    adapter._processes[0].terminate()
    adapter._processes[0].join()
    with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
        adapter.get_state()
    assert adapter._closed and all(not p.is_alive() for p in adapter._processes)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Maze constructor bakes a GL scene; the CPU/software-GL gate runs on Linux"
)
def test_public_chunks_and_early_stop(monkeypatch):
    import stable_envs.envs.ogbench.maze as module

    local = se.make("PointMazeLargeNavigate", num_envs=3, device="cpu", seed=17)
    with monkeypatch.context() as patch:
        patch.setattr(module, "OGBMazeAdapter", partial(OGBMazeProcessAdapter, processes=2))
        remote = se.make("PointMazeLargeNavigate", num_envs=3, device="cpu", seed=17)
    try:
        initial = local.get_state()
        # An already reached goal produces an early stop on only the first lane.
        initial["root", "position"][0] = initial["task", "goal_pos"][0, :2]
        initial["root", "velocity"][0].zero_()
        for horizon in (1, 8):
            actions = np.zeros((3, 2) if horizon == 1 else (3, horizon, 2), np.float64)
            for states in (None, slice(None), slice(None, None, -2), slice(9, 9)):
                for env in (local, remote):
                    env.set_state(initial)
                expected = local.step(actions, states=states, frames=None)
                actual = remote.step(actions, states=states, frames=None)
                assert_same_state(actual, expected)
                if horizon == 8:
                    assert actual["valid"][0].sum().item() == 1
                assert remote._simulator._pending is None
    finally:
        local.close()
        remote.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("maze", ["large", "medium"])
def test_point_lanes_step_as_the_source_step_does(dtype, maze):
    """Point lanes step in one loop and skip MuJoCo for a resting ball; every leaf equals the source step."""
    lanes, steps = 16, 40
    batched, source = OGBMazeAdapter(lanes, seed=23, maze=maze), OGBMazeAdapter(lanes, seed=23, maze=maze)
    source._point = False
    try:
        assert batched._point and batched.process_static_leaves and batched._rest is not None
        assert_same_state(batched.get_state(), source.get_state())
        state = batched.get_state()
        # Distances to the goal on both sides of the success tolerance, including exactly on it.
        offsets = torch.zeros(lanes, 2)
        offsets[:6] = torch.tensor([[0.0, 0.0], [0.6, 0.8], [1.0, 0.0], [0.0, -1.0000001], [3.0, 0.0], [0.7, 0.7]])
        state["root", "position"] = state["task", "goal_pos"][:, :2] + offsets.to(torch.float64)
        state["root", "velocity"].zero_()
        for adapter in (batched, source):
            adapter.set_state(state)
        actions = np.random.default_rng(5).uniform(-1, 1, (lanes, steps, 2)).astype(dtype)
        actions[:, 0] = 0
        resting = []
        for t in range(steps):
            mask = None if t < 2 else torch.tensor([t % (lane % 4 + 1) == 0 for lane in range(lanes)])
            prev = np.stack([env.data.qpos for env in batched.envs])
            resting.append(_resting_lanes(batched._rest, batched.envs, prev + 0.2 * actions[:, t]))
            assert_same_state(batched.step(actions[:, t], active=mask), source.step(actions[:, t], active=mask))
        resting = np.stack(resting)
        assert resting.any() and not resting.all()
        assert_same_state(
            batched.step(actions[:, 1], capture_state=False), source.step(actions[:, 1], capture_state=False)
        )
        assert_same_state(batched.get_state(), source.get_state())
        assert_same_state(batched.reset(7), source.reset(7))
    finally:
        batched.close()
        source.close()
