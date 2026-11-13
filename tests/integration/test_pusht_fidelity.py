"""PushT batched physics against the stable-worldmodel source over 200 steps of a scripted pushing policy.

The policy runs closed loop on the source. Its actions are replayed on stable-envs from the same reset, and
the T-block poses and success flags are compared in every episode, also where the source block passes
through a wall.
"""

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.swm
pytest.importorskip("pygame")
try:
    if version("pymunk") != "7.2.0":
        pytest.skip("PushT requires pymunk==7.2.0", allow_module_level=True)
except PackageNotFoundError:
    pytest.skip("PushT requires pymunk", allow_module_level=True)

import stable_envs as se  # noqa: E402

EPISODES, STEPS, SEED = 48, 200, 1000


def _policy(obs, goal, t, previous):
    """Aim past the block centre toward the goal with a slow lateral sweep, low-pass filtered."""
    agent, block = obs[:2], obs[2:4]
    toward = goal[2:4] - block
    norm = np.linalg.norm(toward)
    u = toward / norm if norm > 1e-6 else np.array([1.0, 0.0])
    target = block + 30 * u + np.array([-u[1], u[0]]) * 40 * np.sin(0.08 * t)
    return 0.6 * previous + 0.4 * np.clip((target - agent) / 100, -1, 1)


def _source_episodes():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    actions = np.zeros((EPISODES, STEPS, 2), np.float32)
    poses = np.zeros((EPISODES, STEPS + 1, 7))
    success = np.zeros((EPISODES, STEPS), bool)
    for lane in range(EPISODES):
        env = gym.make("swm/PushT-v1", resolution=64)
        obs, _ = env.reset(seed=SEED + lane)
        goal = np.asarray(env.unwrapped.goal_state, np.float64)
        poses[lane, 0] = obs["state"]
        action = np.zeros(2)
        for t in range(STEPS):
            action = _policy(obs["state"], goal, t, action)
            actions[lane, t] = action
            obs, _, success[lane, t], *_ = env.step(actions[lane, t])
            poses[lane, t + 1] = obs["state"]
        env.close()
    return actions, poses, success


@pytest.fixture(scope="module")
def source():
    return _source_episodes()


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices())
def test_scripted_pushing_tracks_the_source(source, device):
    actions, poses, success = source
    with se.make("PushT", num_envs=EPISODES, size=64, device=device) as env:
        state = env.reset(seed=SEED)
        np.testing.assert_array_equal(state["current", "observation"].cpu().numpy(), poses[:, 0])
        ours, done = [poses[:, 0]], []
        sequence = torch.as_tensor(actions, device=device)
        for t in range(STEPS):
            out = env.step(sequence[:, t], frames=None)
            ours.append(out["state"]["current", "observation"].cpu().numpy())
            done.append(out["terminated"].cpu().numpy().reshape(EPISODES))
    ours, done = np.stack(ours, 1), np.stack(done, 1)
    gap = np.linalg.norm(ours[:, :, 2:4] - poses[:, :, 2:4], axis=2)
    outside = ((poses[:, :, 2:4] < 0) | (poses[:, :, 2:4] > 512)).any(axis=(1, 2))
    # The source block leaves the board in 16 of the 48 episodes; the port follows it through the wall.
    assert outside.sum() >= 8
    np.testing.assert_array_equal(done, success)
    if device == "cpu":
        # Measured (Numba, float64): the largest gap in any episode at any step was 3.3e-8 px.
        assert gap.max() < 1e-6
    else:
        # Measured (Triton, float32): a median gap of 0.0004 px at step 200, and 39 of 48 blocks within 1 px.
        # float32 rounding grows chaotically where the block is pressed hard into or through a wall.
        assert np.median(gap[:, -1]) < 0.01
        assert (gap[:, -1] < 1.0).mean() >= 0.75
