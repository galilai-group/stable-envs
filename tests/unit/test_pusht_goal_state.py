"""``PushT.goal_state`` draws the goal image of the source reset byte for byte."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pymunk")
torch = pytest.importorskip("torch")
pytest.importorskip("pygame")
pytest.importorskip("cv2")

import stable_envs as snv

pytestmark = pytest.mark.unit


def _source_goals(env):
    return np.stack([owner.env._goal for owner in env._simulator.envs])


@pytest.mark.parametrize(("num_envs", "seed"), [(4, 0), (4, 11), (16, 3)])
def test_goal_frames_equal_the_source_goal_images(num_envs, seed):
    env = snv.PushT(num_envs=num_envs, device="cpu")
    try:
        state = env.reset(seed=seed)
        want = _source_goals(env)
        np.testing.assert_array_equal(env.render(env.goal_state(state)).numpy(), want)
        later = env.step(torch.full((num_envs, 5, 2), 0.4))["state"]
        np.testing.assert_array_equal(env.render(env.goal_state(later)).numpy(), want)
    finally:
        env.close()


def test_goal_state_leaves_the_state_and_the_env_unchanged():
    env = snv.PushT(num_envs=3, device="cpu")
    try:
        env.reset(seed=1)
        state = env.step(torch.full((3, 2), 0.5))["state"]
        before, live = state.clone(), env.get_state()
        goal = env.goal_state(state)
        for a, b in ((state, before), (env.get_state(), live)):
            assert all(torch.equal(x, y) for x, y in zip(a.values(True, True), b.values(True, True)))
        position = snv.get_field(goal, "current/body/block/position")
        target = snv.get_field(goal, "anchor/goal/value")[:, 2:4]
        assert torch.allclose(position, target, atol=5.0)
    finally:
        env.close()


def test_goal_state_of_rows_equals_the_rows_of_goal_state():
    env = snv.PushT(num_envs=4, device="cpu")
    try:
        env.reset(seed=5)
        state = env.step(torch.full((4, 2), -0.3))["state"]
        full, rows = env.goal_state(state), torch.tensor([3, 0])
        part = env.goal_state(state[rows])
        for key, value in part.items(True, True):
            assert torch.equal(value, full[key][rows]), key
    finally:
        env.close()
