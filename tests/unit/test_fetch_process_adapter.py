"""Full Fetch State/lifecycle parity for persistent process and hybrid owners."""

from functools import partial

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv
import stable_envs.envs._fetch_semantics as semantics
from stable_envs._adapters.fetch_pickandplace import FetchAdapter
from stable_envs._adapters.fetch_process import FetchProcessAdapter


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert a[key].dtype == b[key].dtype, key
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


@pytest.mark.parametrize("task", ["Reach", "Push", "Slide", "PickAndPlace"])
@pytest.mark.parametrize("suffix", ["", "Dense", "Dict"])
@pytest.mark.parametrize("threads", [1, 2])
def test_public_contract_and_variations(monkeypatch, task, suffix, threads):
    name = "Fetch" + task + suffix
    reference = snv.make(name, num_envs=5, seed=17, device="cpu", backend="opengl")
    with monkeypatch.context() as patch:
        patch.setattr(
            semantics, "FetchAdapter", partial(FetchProcessAdapter, processes=2, threads_per_process=threads)
        )
        candidate = snv.make(name, num_envs=5, seed=17, device="cpu", backend="opengl")
    try:
        equal(reference.get_state(), candidate.get_state())
        actions = np.random.default_rng(31).uniform(-1, 1, (5, 6, 4)).astype(np.float64)
        initial = reference.get_state()
        for states in (slice(None), slice(None, None, -2), None):
            reference.set_state(initial)
            candidate.set_state(initial)
            expected = reference.step(actions, states=states, frames=None)
            actual = candidate.step(actions, states=states, frames=None)
            equal(expected, actual)
        frozen = actual.clone()
        candidate.reset(seed=42)
        equal(actual, frozen)
        # Checkpoint/RNG replay includes seed=None continuation, not just seeded resets.
        for env in (reference, candidate):
            env.set_state(initial)
        equal(reference.reset(), candidate.reset())
        for env in (reference, candidate):
            env.set_state(initial)
            env._simulator._set_episode_limits(np.arange(1, 6))
        expected = reference.step(actions, frames=None)
        actual = candidate.step(actions, frames=None)
        equal(expected, actual)
        assert actual["valid"].sum(1).tolist() == [1, 2, 3, 4, 5]
        for env in (reference, candidate):
            env._simulator._set_episode_limits(np.full(5, 50))
        variation = {"table.color": np.array([0.2, 0.7, 0.4])}
        if task != "Reach":
            variation.update(
                {
                    "object.color": np.array([0.8, 0.1, 0.3]),
                    "block.mass": np.array([0.9]),
                    "block.angle": np.array([0.3]),
                    "block.start_position": np.array([1.3, 0.7]),
                }
            )
        for seed, values in ((42, variation), (17, {})):
            expected = reference.reset(seed=seed, options={"variation_values": values})
            actual = candidate.reset(seed=seed, options={"variation_values": values})
            equal(expected, actual)
            assert torch.equal(reference.get_style(), candidate.get_style())
            equal(
                reference.step(actions.astype(np.float32), frames=None),
                candidate.step(actions.astype(np.float32), frames=None),
            )
    finally:
        reference.close()
        candidate.close()
    assert all(not p.is_alive() for p in candidate._simulator._processes)
    candidate.close()


def test_invalid_state_and_actions_do_not_mutate():
    adapter = FetchProcessAdapter(5, task="Push", processes=2, threads_per_process=2)
    try:
        initial = adapter.get_state()
        invalid = initial.clone()
        invalid["qpos"][4, 0] = float("nan")  # Invalid in the last shard.
        with pytest.raises(ValueError):
            adapter.set_state(invalid)
        equal(initial, adapter.get_state())
        with pytest.raises(ValueError):
            adapter.step(np.zeros((4, 4)))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((5, 4)), active=np.ones(4, dtype=bool))
        equal(initial, adapter.get_state())
        equal(initial, adapter.step(np.zeros((5, 4)), active=np.zeros(5, dtype=bool))["state"])
        owned = adapter.get_state()
        owned["qpos"].fill_(99)
        equal(initial, adapter.get_state())
    finally:
        adapter.close()


def test_worker_death_fails_closed_and_reaps_every_owner():
    adapter = FetchProcessAdapter(3, task="Push", processes=2)
    adapter._processes[0].terminate()
    adapter._processes[0].join()
    with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
        adapter.get_state()
    assert adapter._closed and all(not p.is_alive() for p in adapter._processes)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.step(np.zeros((3, 4)))
    adapter.close()


def test_inactive_lanes_match_local_owner_before_first_step():
    local, remote = FetchAdapter(5, task="Push"), FetchProcessAdapter(5, task="Push", processes=2)
    try:
        actions = np.zeros((5, 4), dtype=np.float32)
        equal(
            local.step(actions, active=np.array([1, 0, 0, 0, 1], dtype=bool)),
            remote.step(actions, active=np.array([1, 0, 0, 0, 1], dtype=bool)),
        )
    finally:
        local.close()
        remote.close()
