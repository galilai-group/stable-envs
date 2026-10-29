import hashlib

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.first_family import ReacherAdapter  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402


@pytest.mark.parametrize("backend", ["auto", "native"])
def test_public_cpu_reacher_keeps_native_physics(backend):
    import stable_envs as se

    reference = ReacherAdapter(1, "cpu", 7)
    try:
        with se.DMCReacher(num_envs=1, device="cpu", simulation_backend=backend) as env:
            assert env.simulation_backend == "native"
            assert env.execution_mode == "DELEGATED_CPU"
            checkpoint = env.reset(seed=7)
            reference.reset(7)
            action = torch.tensor([[0.2, -0.4]])
            expected = reference.step(action)
            actual = env.step(action, frames=None)
            for key in expected.keys():
                assert_same_state(actual[key], expected[key])
            env.set_state(checkpoint)
            assert_same_state(env.step(action, frames=None), actual)
    finally:
        reference.close()


@pytest.fixture
def pair():
    pytest.importorskip("mujoco.mjx")
    from stable_envs.envs.dmcontrol.simulation.reacher_mjx import ReacherMJXAdapter

    reference = ReacherAdapter(3, "cpu", 7)
    env = ReacherMJXAdapter(3, "cpu", 7)
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
    action = torch.zeros(3, 2)
    for target_x, reward in ((0.27, 0.0), (0.26, 2.0)):
        state["task", "target_pos"][:] = torch.tensor([target_x, 0, 0.01], dtype=torch.float64)
        for item in pair:
            item.set_state(state)
        expected = reference.step(action)["reward"]
        assert torch.equal(expected, torch.full((3,), reward, dtype=torch.float64))
        assert torch.equal(env.step(action)["reward"], expected)
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


def test_warm_step_does_not_hash_static_model(pair, monkeypatch):
    _, env = pair
    action = torch.zeros(3, 2)
    env.step(action)

    sha256 = hashlib.sha256

    def hash_empty_metadata(value=b"", **kwargs):
        assert not value, "Static model hashing in a warmed Reacher step"
        return sha256(value, **kwargs)

    monkeypatch.setattr("hashlib.sha256", hash_empty_metadata)
    env.step(action)


def test_density_changes_after_compilation_and_checkpoint_restore(pair):
    reference, env = pair
    action = torch.tensor([[0.2, -0.4], [1.0, -1.0], [-0.3, 0.6]])
    checkpoint = reference.get_state()
    env.step(action)
    for item in pair:
        item.configure_physics(np.array([[1500, 500], [500, 1500], [1150, 850]]))
        item.reset(7)
    for state in (None, checkpoint):
        if state is not None:
            for item in pair:
                item.set_state(state)
        for _ in range(4):
            expected, actual = reference.step(action), env.step(action)
            for key in ("qpos", "qvel"):
                torch.testing.assert_close(actual["state", key], expected["state", key], rtol=0, atol=3e-6)
