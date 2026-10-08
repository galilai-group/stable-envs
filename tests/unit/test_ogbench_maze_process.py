"""Persistent Maze ownership, atomic validation, RNG isolation and lookahead."""

import sys
from functools import partial

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as se
from stable_envs._adapters.ogbench.maze import OGBMazeAdapter
from stable_envs._adapters.ogbench.maze_process import OGBMazeProcessAdapter
from tests.unit.test_ogbench_state_export import equal


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_state_rng_outputs_and_inactive_lanes(dtype):
    local, remote = OGBMazeAdapter(5, seed=17), OGBMazeProcessAdapter(5, seed=17, processes=2)
    try:
        equal(local.get_state(), remote.get_state())
        initial = local.get_state()
        initial["rng", "global_has_gauss"][4] = True
        initial["rng", "global_cached_gaussian"][4] = 0.125
        initial["rng", "env_has_uint32"][4] = 1
        initial["rng", "env_uinteger"][4] = 2**32 - 1
        for adapter in (local, remote):
            adapter.set_state(initial)
        actions = np.random.default_rng(31).uniform(-0.5, 0.5, (5, 8, 2)).astype(dtype)
        held = remote.get_state()
        for t in range(8):
            mask = None if t == 0 else torch.tensor([True, False, True, False, True])
            equal(local.step(actions[:, t], active=mask), remote.step(actions[:, t], active=mask))
        equal(local.step(actions[:, 0], capture_state=False), remote.step(actions[:, 0], capture_state=False))
        equal(local.get_state(), remote.get_state())
        equal(held, initial)
        for adapter in (local, remote):
            adapter.set_state(initial)
        equal(local.reset(), remote.reset())
        equal(local.reset(42), remote.reset(42))
        edited = remote.get_state()
        edited["root", "position"].zero_()
        equal(local.get_state(), remote.get_state())
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
        equal(initial, adapter.get_state())
        for call in (lambda: adapter.reset(-1), lambda: adapter.step(np.zeros((4, 2))),
                     lambda: adapter.step(np.zeros((5, 2)), active=np.ones(5))):
            with pytest.raises(ValueError):
                call()
            equal(initial, adapter.get_state())
        adapter.step(np.zeros((5, 2)))
        adapter.set_state(initial)
        equal(initial, adapter.get_state())
    finally:
        adapter.close()


def test_pending_request_exclusion_owned_output_and_close():
    adapter = OGBMazeProcessAdapter(3, seed=17, processes=2)
    try:
        initial = adapter.get_state()
        adapter.submit_step(np.zeros((3, 2)), capture_state=False)
        for call in (adapter.get_state, lambda: adapter.set_state(initial), adapter.reset,
                     lambda: adapter.submit_step(np.zeros((3, 2)))):
            with pytest.raises(RuntimeError, match="in flight"):
                call()
        result = adapter.receive_step()
        assert "state" not in result.keys()
        held = result.clone()
        adapter.step(np.ones((3, 2)))
        equal(result, held)
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


@pytest.mark.skipif(sys.platform == "darwin", reason="Maze constructor bakes a GL scene; CPU/software-GL gate runs on Oscar")
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
                equal(actual, expected)
                if horizon == 8:
                    assert actual["valid"][0].sum().item() == 1
                assert remote._simulator._pending is None
    finally:
        local.close()
        remote.close()
