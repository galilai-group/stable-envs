"""Bounded Acrobot ownership, source continuation, lifecycle and public slices."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch

import stable_envs as se
import stable_envs.envs.classic_control.semantics as semantics
from stable_envs.envs.classic_control.simulation.local import ClassicControlAdapter
from stable_envs.envs.classic_control.simulation.process import AcrobotProcessAdapter
from tests.unit.test_classic_state_export import equal


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_state_task_rng_variation_and_inactive_lanes(dtype):
    local = ClassicControlAdapter("acrobot", 5)
    remote = AcrobotProcessAdapter("acrobot", 5, seed=17, processes=2)
    try:
        equal(local.reset(17), remote.get_state())
        initial = local.get_state()
        initial["rng", "has_uint32"][-1] = 1
        initial["rng", "uinteger"][-1] = 2**32 - 1
        values = {"link_mass_1": np.linspace(0.5, 1.5, 5)}
        for adapter in (local, remote):
            adapter.configure(values)
            adapter.set_state(initial)
        held = remote.get_state()
        actions = np.random.default_rng(31).integers(-3, 3, (5, 8, 1)).astype(dtype)
        for t in range(8):
            mask = None if t == 0 else torch.tensor([True, False, True, False, True])
            equal(local.step(actions[:, t], active=mask), remote.step(actions[:, t], active=mask))
        equal(local.step(actions[:, 0]).exclude("state"), remote.step(actions[:, 0], capture_state=False))
        equal(local.get_state(), remote.get_state())
        equal(held, initial)
        equal(local.reset(), remote.reset())
        equal(local.reset(42, {"low": -0.2, "high": 0.3}), remote.reset(42, {"low": -0.2, "high": 0.3}))
        changed = remote.get_state()
        changed["theta1"].zero_()
        equal(local.get_state(), remote.get_state())
    finally:
        local.close()
        remote.close()
    assert all(not p.is_alive() for p in remote._processes)


def test_invalid_last_lane_does_not_mutate_other_owners():
    remote = AcrobotProcessAdapter("acrobot", 5, seed=17, processes=2)
    try:
        initial = remote.get_state()
        invalid = initial.clone()
        invalid["theta1"][0] += 0.25
        invalid["rng", "state"][-1].fill_(-1)
        with pytest.raises((ValueError, OverflowError)):
            remote.set_state(invalid)
        equal(initial, remote.get_state())
        bad = initial.clone()
        bad["dtheta1"][-1] = float("nan")
        with pytest.raises(ValueError):
            remote.set_state(bad)
        equal(initial, remote.get_state())
        for actions, error in [
            (np.ones((5, 1), np.float32), TypeError),
            (np.full((5, 1), 3), IndexError),
            (np.ones((4, 1)), ValueError),
        ]:
            with pytest.raises(error):
                remote.step(actions)
            equal(initial, remote.get_state())
        remote.step(np.ones((5, 1), np.int64))
        remote.set_state(initial)
        equal(initial, remote.get_state())
        assert not remote._closed
    finally:
        remote.close()


def test_one_request_owned_outputs_and_close():
    remote = AcrobotProcessAdapter("acrobot", 3, processes=2)
    initial = remote.get_state()
    try:
        remote.submit_step(np.ones((3, 1), np.int64), capture_state=False)
        for call in (
            remote.get_state,
            remote.reset,
            lambda: remote.set_state(initial),
            lambda: remote.submit_step(np.ones((3, 1), np.int64)),
        ):
            with pytest.raises(RuntimeError, match="in flight"):
                call()
        result = remote.receive_step()
        assert "state" not in result.keys()
        held = result.clone()
        remote.step(np.zeros((3, 1), np.int64))
        equal(result, held)
        remote.submit_step(np.ones((3, 1), np.int64))
    finally:
        remote.close()
    assert all(not p.is_alive() for p in remote._processes)
    with pytest.raises(RuntimeError, match="closed"):
        remote.get_state()


def test_worker_death_fails_closed():
    remote = AcrobotProcessAdapter("acrobot", 3, processes=2)
    remote._processes[0].terminate()
    remote._processes[0].join()
    with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
        remote.get_state()
    assert remote._closed and all(not p.is_alive() for p in remote._processes)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_chunks_slices_early_stop_and_current_render(pipeline, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    local = se.make("Acrobot", num_envs=4, seed=17, device=device)
    with patch.object(semantics, "ClassicControlAdapter", partial(AcrobotProcessAdapter, processes=2)):
        remote = se.make("Acrobot", num_envs=4, seed=17, device=device)
    if pipeline:
        remote._step_pipeline = lambda: remote._simulator
    try:
        initial = local.get_state()
        initial["lifecycle", "elapsed_steps"][:, 0] = torch.tensor([0, 499, 500, 4], device=device)
        initial["theta1"][3] = np.pi
        initial["rng", "has_uint32"][-1] = 1
        initial["rng", "uinteger"][-1] = 2**32 - 1
        actions = np.random.default_rng(31).integers(0, 3, (4, 8, 1))
        for horizon in (1, 8):
            a = actions[:, 0, 0] if horizon == 1 else actions
            for states in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(10, 12)):
                for env in (local, remote):
                    env.set_state(initial)
                actual = remote.step(a, states=states, frames=states)
                expected = local.step(a, states=states, frames=states)
                equal(actual, expected)
                if horizon == 8:
                    assert actual["valid"][1:].sum(1).tolist() == [1, 1, 1]
                torch.testing.assert_close(local.render(), remote.render(), rtol=0, atol=0)
                held = actual.clone()
                remote.reset(42)
                equal(actual, held)
        for env in (local, remote):
            env.set_state(initial)
        equal(local.reset(), remote.reset())
        for env in (local, remote):
            env.set_variation({"physics.link_mass_1": [1.3]})
            env.reset(31)
        equal(
            local.step(torch.as_tensor(actions, device=device)), remote.step(torch.as_tensor(actions, device=device))
        )
    finally:
        local.close()
        remote.close()


def test_candidate_does_not_accept_other_families():
    with pytest.raises(ValueError, match="bounded to Acrobot"):
        AcrobotProcessAdapter("cartpole", 4, processes=2)
