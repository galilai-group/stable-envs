"""Reacher uses the shared Warp runtime with native targets, resets and Variation."""

import math
from contextlib import closing

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.dmcontrol.simulation.first_family import ReacherAdapter
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_reacher_matches_native_for_100_calls_with_variation_and_resets():
    with (
        closing(ReacherAdapter(3, "cpu", 7)) as reference,
        se.make("DMCReacher", num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env,
    ):
        reference.configure_physics(np.array([[500, 1500], [1500, 500], [850, 1150]]))
        initial = reference.reset(7)
        limit = math.ceil(reference.envs[0]._env._step_limit)
        # Reset in the second control round, reset in the first, and continue.
        initial["lifecycle", "step_count"][:, 0] = torch.tensor([limit - 1, limit, 6])
        initial["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False])
        initial["task", "cumulative_reward"][:, 0] = torch.tensor([11.0, 17.0, 23.0])
        reference.set_state(initial)
        env.set_state(initial.cuda())
        assert_same_state(env.get_state().cpu(), initial)

        # Match the existing Reacher native/MJX position, velocity and observation
        # bounds, but advance 100 public calls without intermediate restoration.
        for action in np.random.default_rng(43).uniform(-1, 1, (100, 3, 2)):
            expected = reference.step(action)
            actual = env.step(torch.as_tensor(action, device="cuda"), states=None, frames=None).cpu()
            torch.testing.assert_close(actual["state", "qpos"], expected["state", "qpos"], rtol=0, atol=1e-5)
            torch.testing.assert_close(actual["state", "qvel"], expected["state", "qvel"], rtol=0, atol=3e-5)
            torch.testing.assert_close(actual["observation"], expected["observation"], rtol=0, atol=3e-5)
            for key in ("reward", "score", "terminated", "truncated"):
                assert_same_state(actual[key], expected[key])
            for key in ("rng", "lifecycle", "time", "physics", ("task", "target_pos"), ("task", "cumulative_reward")):
                assert_same_state(actual["state"][key], expected["state"][key])
        assert not torch.equal(actual["state", "task", "target_pos"][:2], initial["task", "target_pos"][:2])
        assert_same_state(actual["state", "task", "target_pos"][2], initial["task", "target_pos"][2])


def test_reacher_target_restore_preserves_sparse_rewards():
    with (
        closing(ReacherAdapter(3, "cpu", 7)) as reference,
        se.make("DMCReacher", num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env,
    ):
        state = reference.get_state()
        state["qpos"].zero_()
        state["qvel"].zero_()
        state["warmstart"].zero_()
        targets = torch.tensor([[0.26, 0, 0.01], [0.27, 0, 0.01], [0.24, 0.1, 0.01]], dtype=torch.float64)
        action = torch.zeros(3, 2, dtype=torch.float64)
        for order in ((0, 1, 2), (1, 2, 0)):
            state["task", "target_pos"][:] = targets[list(order)]
            reference.set_state(state)
            env.set_state(state.cuda())
            expected = reference.step(action)
            actual = env.step(action.cuda(), states=None, frames=None).cpu()
            assert_same_state(
                expected["reward"], torch.tensor([2.0 if lane == 0 else 0.0 for lane in order], dtype=torch.float64)
            )
            assert_same_state(actual["reward"], expected["reward"])
            torch.testing.assert_close(actual["observation"], expected["observation"], rtol=0, atol=3e-6)
            assert_same_state(actual["state", "task", "target_pos"], state["task", "target_pos"])

        held = env.get_state()
        invalid = held.clone()
        invalid["task", "target_pos"][0, 0] += 0.1
        invalid["task", "target_pos"][2, 0] = float("nan")
        with pytest.raises(ValueError):
            env.set_state(invalid)
        assert_same_state(env.get_state(), held)


def test_reacher_replays_100_public_calls():
    with se.make("DMCReacher", num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        env.reset(seed=31)
        actions = torch.as_tensor(np.random.default_rng(53).uniform(-1, 1, (105, 3, 2)), device="cuda")
        for action in actions[:5]:
            env.step(action, states=None, frames=None)
        checkpoint = env.get_state()
        expected = [env.step(action, states=None, frames=None).cpu() for action in actions[5:]]
        env.reset(seed=77)
        env.set_state(checkpoint)
        assert_same_state(env.get_state(), checkpoint)
        for action, want in zip(actions[5:], expected, strict=True):
            actual = env.step(action, states=None, frames=None).cpu()
            assert_same_state(actual, want)
