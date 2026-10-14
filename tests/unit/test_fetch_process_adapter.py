"""Process-owned Fetch simulators match the in-process owner on State, lifecycle and variation resets."""

from functools import partial

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv
import stable_envs.envs.fetch.semantics as semantics
from stable_envs.envs.fetch.simulation.adapter import FetchAdapter
from stable_envs.envs.fetch.simulation.process import FetchProcessAdapter
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize(
    "task,suffix,threads",
    [
        ("Reach", "", 1),
        ("Reach", "Dict", 2),
        ("Push", "Dense", 1),
        ("Slide", "", 2),
        ("PickAndPlace", "Dict", 1),
        ("PickAndPlace", "Dense", 2),
    ],
)
def test_process_owner_matches_local_owner_through_the_public_env(monkeypatch, task, suffix, threads):
    name = "Fetch" + task + suffix
    reference = snv.make(name, num_envs=5, seed=17, device="cpu", backend="opengl")
    with monkeypatch.context() as patch:
        patch.setattr(
            semantics, "FetchAdapter", partial(FetchProcessAdapter, processes=2, threads_per_process=threads)
        )
        candidate = snv.make(name, num_envs=5, seed=17, device="cpu", backend="opengl")
    try:
        assert_same_state(reference.get_state(), candidate.get_state())
        actions = np.random.default_rng(31).uniform(-1, 1, (5, 6, 4)).astype(np.float64)
        initial = reference.get_state()
        for states in (slice(None), slice(None, None, -2), None):
            reference.set_state(initial)
            candidate.set_state(initial)
            expected = reference.step(actions, states=states, frames=None)
            actual = candidate.step(actions, states=states, frames=None)
            assert_same_state(expected, actual)
        frozen = actual.clone()
        candidate.reset(seed=42)
        assert_same_state(actual, frozen)
        # Checkpoint/RNG replay includes seed=None continuation, not just seeded resets.
        for env in (reference, candidate):
            env.set_state(initial)
        assert_same_state(reference.reset(), candidate.reset())
        for env in (reference, candidate):
            env.set_state(initial)
            env._simulator._set_episode_limits(np.arange(1, 6))
        expected = reference.step(actions, frames=None)
        actual = candidate.step(actions, frames=None)
        assert_same_state(expected, actual)
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
            assert_same_state(expected, actual)
            assert torch.equal(reference.get_style(), candidate.get_style())
            assert_same_state(
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
        assert_same_state(initial, adapter.get_state())
        with pytest.raises(ValueError):
            adapter.step(np.zeros((4, 4)))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((5, 4)), active=np.ones(4, dtype=bool))
        assert_same_state(initial, adapter.get_state())
        assert_same_state(initial, adapter.step(np.zeros((5, 4)), active=np.zeros(5, dtype=bool))["state"])
        owned = adapter.get_state()
        owned["qpos"].fill_(99)
        assert_same_state(initial, adapter.get_state())
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
        assert_same_state(
            local.step(actions, active=np.array([1, 0, 0, 0, 1], dtype=bool)),
            remote.step(actions, active=np.array([1, 0, 0, 0, 1], dtype=bool)),
        )
    finally:
        local.close()
        remote.close()


def test_inflight_exclusion_ownership_and_close():
    local, remote = FetchAdapter(5, task="Push"), FetchProcessAdapter(5, task="Push", processes=2)
    try:
        checkpoint = remote.get_state()
        actions = np.zeros((5, 4), dtype=np.float64)
        remote.submit_step(actions)
        for call in (
            remote.get_state,
            lambda: remote.set_state(checkpoint),
            lambda: remote.submit_step(actions),
            lambda: remote.reset(42),
        ):
            with pytest.raises(RuntimeError, match="in flight"):
                call()
        first = remote.receive_step()
        assert_same_state(first, local.step(actions))
        held = first.clone()
        remote.submit_step(actions)
        assert_same_state(remote.receive_step(), local.step(actions))
        assert_same_state(first, held)
        with pytest.raises(RuntimeError, match="No Fetch step"):
            remote.receive_step()
        remote.submit_step(actions)
        remote.close()
        assert all(not p.is_alive() for p in remote._processes)
    finally:
        local.close()
        remote.close()


def test_public_pipeline_renderer_failure_reaps_workers(monkeypatch):
    monkeypatch.setattr(semantics, "FetchAdapter", partial(FetchProcessAdapter, processes=2))
    with snv.make("FetchPush", num_envs=5, device="cpu") as env:

        def fail(state):
            assert env._simulator._pending is not None
            raise RuntimeError("render failed")

        monkeypatch.setattr(env, "render", fail)
        with pytest.raises(RuntimeError, match="render failed"):
            env.step(np.zeros((5, 8, 4)), frames=slice(None))
        assert env._simulator._closed
        assert all(not p.is_alive() for p in env._simulator._processes)


def test_workers_finish_next_step_while_parent_consumes_owned_frame_state(monkeypatch):
    monkeypatch.setattr(semantics, "FetchAdapter", partial(FetchProcessAdapter, processes=2))
    with snv.make("FetchPush", num_envs=5, device="cpu") as env:
        overlapped = []

        def render(state):
            elapsed = state["lifecycle", "elapsed_steps"].clone()
            owner = env._simulator
            if owner._pending is not None:
                # Waiting only for readiness does not consume acknowledgments.
                assert all(conn.poll(10) for conn in owner._connections)
                shared = owner._views["state", ("lifecycle", "elapsed_steps")]
                assert np.array_equal(shared, elapsed.numpy() + 1)
                assert torch.equal(elapsed, state["lifecycle", "elapsed_steps"])
                overlapped.append(int(elapsed[0, 0]))
            return elapsed.to(torch.uint8).reshape(5, 1, 1, 1).expand(5, 2, 2, 3).clone()

        monkeypatch.setattr(env, "render", render)
        out = env.step(np.zeros((5, 8, 4)), frames=slice(None))
        assert overlapped == list(range(1, 8))
        assert out["frames"][0, :, 0, 0, 0].tolist() == list(range(1, 9))
        assert env._simulator._pending is None


def test_inflight_worker_death_fails_closed():
    adapter = FetchProcessAdapter(5, task="Push", processes=2)
    adapter._processes[0].terminate()
    adapter._processes[0].join()
    try:
        with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
            adapter.submit_step(np.zeros((5, 4)))
            adapter.receive_step()
        assert adapter._closed
        assert all(not p.is_alive() for p in adapter._processes)
    finally:
        adapter.close()
