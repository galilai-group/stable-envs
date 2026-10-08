"""Exact task/State equivalence and failure ownership for DMC process shards."""

import numpy as np
import pytest
import torch

from stable_envs._adapters.dmc.process import DMCProcessAdapter, make_local

DOMAINS = ["acrobot", "pendulum", "cartpole", "reacher", "cheetah", "hopper", "walker",
           "ball_in_cup", "finger", "manipulator", "quadruped", "humanoid"]


def equal(left, right):
    assert left.batch_size == right.batch_size
    assert set(left.keys(True, True)) == set(right.keys(True, True))
    for key in left.keys(True, True):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True, msg=str(key))


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_sharded_tasks_rng_lifecycle_and_owned_outputs(domain, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA gate runs on Oscar")
    local = make_local(domain, 5, device, 17)
    remote = DMCProcessAdapter(domain, 5, device, 17, processes=2)
    try:
        initial = local.get_state()
        equal(initial, remote.get_state())
        initial["rng", "has_gauss"][3] = True
        initial["rng", "cached_gaussian"][3] = 0.125
        initial["lifecycle", "reset_next_step"][3] = True
        initial["lifecycle", "step_count"][4] = int(local.envs[0]._env._step_limit) - 1
        actions = np.random.default_rng(31).uniform(-0.5, 0.5, (5, 8, remote.action_dim))
        for owner in (local, remote):
            owner.set_state(initial)
        for t in range(8):
            # Both NumPy and device-resident float32 inputs retain source conversion.
            action = actions[:, t] if t % 2 else torch.as_tensor(actions[:, t], dtype=torch.float32, device=device)
            expected = local.step(action)
            remote.submit_step(action, capture_state=t % 2 == 0)
            actual = remote.receive_step()
            if t % 2:
                assert "state" not in actual.keys()
                actual["state"] = remote.get_state()
            equal(expected, actual)
        retained = actual.clone()
        for owner in (local, remote):
            owner.set_state(initial)
        equal(local.reset(), remote.reset())
        equal(local.reset(seed=42), remote.reset(seed=42))
        equal(retained, actual)
        own = remote.get_state()
        snapshot = own.clone()
        own["qpos"].fill_(7)
        equal(snapshot, remote.get_state())
    finally:
        local.close()
        remote.close()
    assert all(not child.is_alive() for child in remote._processes)


@pytest.mark.parametrize("domain,method,width", [
    ("acrobot", "configure_gravity", 3),
    ("humanoid", "configure_gravity", 3),
    ("reacher", "configure_density", 2),
    ("finger", "configure_proximal_density", 1),
    ("manipulator", "configure_upper_arm_density", 1),
])
def test_configuration_preserves_whole_batch_effects(domain, method, width):
    local = make_local(domain, 5, seed=17)
    remote = DMCProcessAdapter(domain, 5, seed=17, processes=2)
    try:
        action = np.full((5, remote.action_dim), 0.3)
        equal(local.step(action), remote.step(action))
        values = np.full((5, width), 1000.0)
        if method == "configure_gravity":
            values[:] = [0, 0, -9.81]
            values[4] = [0.1, 0.2, -8.0]
        else:
            values[4] = 1100.0  # Only the last shard changes.
        for adapter in (local, remote):
            getattr(adapter, method)(values)
        equal(local.get_state(), remote.get_state())
        equal(local.step(action), remote.step(action))
        # Repeating the same density configuration must not reset either owner.
        for adapter in (local, remote):
            getattr(adapter, method)(values)
        equal(local.get_state(), remote.get_state())
    finally:
        local.close()
        remote.close()


def test_invalid_requests_do_not_mutate_and_inflight_requests_cannot_write_buffers():
    owner = DMCProcessAdapter("reacher", 5, processes=2)
    try:
        initial = owner.get_state()
        invalid = initial.clone()
        invalid["qpos"][4, 0] = float("nan")
        with pytest.raises(ValueError):
            owner.set_state(invalid)
        with pytest.raises(ValueError):
            owner.step(np.full((5, 2), np.nan))
        equal(initial, owner.get_state())
        owner.submit_step(np.zeros((5, 2)))
        for call in (lambda: owner.set_state(initial), owner.reset, owner.get_state,
                     lambda: owner.configure_density(np.full((5, 2), 1000))):
            with pytest.raises(RuntimeError, match="in flight"):
                call()
        actual = owner.receive_step()
        assert actual["state"]["time"].min() > initial["time"].max()
        with pytest.raises(RuntimeError, match="No DMC step"):
            owner.receive_step()
    finally:
        owner.close()


def test_worker_death_fails_closed_and_reaps_all_shards():
    owner = DMCProcessAdapter("acrobot", 3, processes=2)
    owner._processes[0].terminate()
    owner._processes[0].join()
    try:
        with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
            owner.get_state()
        assert owner._closed and all(not child.is_alive() for child in owner._processes)
    finally:
        owner.close()
