"""MJX must preserve the registered Reacher task around float32 physics."""

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.first_family import ReacherAdapter  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402


@pytest.fixture(params=["easy", "hard"])
def pair(request):
    pytest.importorskip("mujoco.mjx")
    from stable_envs.envs.dmcontrol.simulation.reacher_mjx import ReacherMJXAdapter

    reference = ReacherAdapter(3, "cpu", 7, request.param)
    env = ReacherMJXAdapter(3, "cpu", 7, request.param)
    if request.param == "hard":
        for item in (reference, env):
            item.configure_physics(np.array([[500, 1500], [1500, 500], [850, 1150]]))
            item.reset(7)
    yield reference, env
    env.close()
    reference.close()


def test_matches_native_dynamics_and_independent_implicit_resets(pair):
    reference, env = pair
    initial = reference.get_state()
    initial["lifecycle", "step_count"][:, 0] = torch.tensor([997, 998, 999])
    for item in pair:
        item.set_state(initial)
    actions = np.random.default_rng(43).uniform(-1, 1, (32, 3, 2))
    for action in actions:
        expected, actual = reference.step(action), env.step(action)
        # Float32 integration accumulates rounding over these 64 physics steps.
        torch.testing.assert_close(actual["state", "qpos"], expected["state", "qpos"], rtol=0, atol=1e-5)
        torch.testing.assert_close(actual["state", "qvel"], expected["state", "qvel"], rtol=0, atol=3e-5)
        torch.testing.assert_close(actual["observation"], expected["observation"], rtol=0, atol=3e-5)
        for key in ("reward", "score", "terminated", "truncated"):
            assert torch.equal(actual[key], expected[key])
        for key in ("rng", "lifecycle", "time", ("task", "target_pos")):
            assert_same_state(actual["state"][key], expected["state"][key])
    assert not torch.equal(actual["state", "task", "target_pos"], initial["task", "target_pos"])


def test_checkpoint_restores_reward_and_rejects_invalid_state_atomically(pair):
    reference, env = pair
    state = reference.get_state()
    state["qpos"].zero_()
    state["qvel"].zero_()
    state["task", "target_pos"][:] = torch.tensor([0.24, 0, 0.01], dtype=torch.float64)
    for item in pair:
        item.set_state(state)
    action = torch.zeros(3, 2)
    assert torch.equal(env.step(action)["reward"], reference.step(action)["reward"])
    assert torch.equal(env.step(action)["reward"], torch.full((3,), 2.0, dtype=torch.float64))
    checkpoint = env.get_state()
    action = torch.tensor([[0.2, -0.4], [1.0, -1.0], [-0.3, 0.6]])
    expected = env.step(action)
    env.set_state(checkpoint)
    assert_same_state(env.step(action), expected)
    held = env.get_state()
    broken = held.clone()
    broken["qvel"][2, 0] = float("nan")
    with pytest.raises(ValueError):
        env.set_state(broken)
    assert_same_state(env.get_state(), held)
    checkpoint["qpos"].fill_(100)
    assert_same_state(env.get_state(), held)
