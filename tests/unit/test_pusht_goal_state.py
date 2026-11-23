"""``PushT.goal_render_state`` draws the goal image of the source reset byte for byte."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pymunk")
torch = pytest.importorskip("torch")
pytest.importorskip("pygame")
pytest.importorskip("cv2")

import stable_envs as snv
from stable_envs.envs.pusht.simulation import numeric

pytestmark = pytest.mark.unit


def _source_goals(env):
    return np.stack([owner.env._goal for owner in env._simulator.envs])


@pytest.mark.parametrize("name", ["PushT", "PushTDiscrete"])
@pytest.mark.parametrize(("num_envs", "seed"), [(4, 0), (4, 11), (16, 3)])
def test_goal_frames_equal_the_source_goal_images(name, num_envs, seed):
    env = snv.make(name, num_envs=num_envs, device="cpu")
    try:
        state = env.reset(seed=seed)
        want = _source_goals(env)
        np.testing.assert_array_equal(env.render(env.goal_render_state(state)).numpy(), want)
        actions = torch.full((num_envs, 5, 2), 3 if name == "PushTDiscrete" else 0.4)
        later = env.step(actions.to(env.action_spec.dtype))["state"]
        np.testing.assert_array_equal(env.render(env.goal_render_state(later)).numpy(), want)
    finally:
        env.close()


def test_a_goal_scene_with_more_segments_than_the_state_keeps_draws_from_the_reset():
    """A row whose goal segment count exceeds ``GOAL_SEGMENTS`` takes its segments from a source reset."""
    env = snv.PushT(num_envs=3, device="cpu")
    try:
        state = env.reset(seed=0)
        snv.get_field(state, "anchor/goal_scene/render_segment_count")[1] = numeric.GOAL_SEGMENTS + 1
        np.testing.assert_array_equal(env.render(env.goal_render_state(state)).numpy(), _source_goals(env))
    finally:
        env.close()


def test_rows_that_end_inside_a_sequence_keep_the_goal_of_their_reset():
    env = snv.PushT(num_envs=3, device="cpu")
    try:
        env.reset(seed=6)
        goal = snv.get_field(env.get_state(), "anchor/goal/value")[0].numpy().copy()
        # Every row starts at its goal, so it succeeds on the first action and holds for the rest of the sequence.
        state = env.reset(seed=6, options={"state": goal, "goal_state": goal})
        want = _source_goals(env)
        out = env.step(torch.zeros(3, 6, 2))
        assert out["valid"][:, 0].all() and not out["valid"][:, 1:].any()
        np.testing.assert_array_equal(env.render(env.goal_render_state(out["state"])).numpy(), want)
        for name in ("anchor/goal/value", "anchor/goal_scene/block/position"):
            assert torch.equal(snv.get_field(out["state"], name), snv.get_field(state, name)), name
    finally:
        env.close()


def test_goal_render_state_is_a_render_state_that_set_state_refuses():
    env = snv.PushT(num_envs=4, device="cpu")
    try:
        env.reset(seed=5)
        state = env.step(torch.full((4, 2), -0.3))["state"]
        before = state.clone()
        goal = env.goal_render_state(state)
        assert all(torch.equal(a, b) for a, b in zip(state.values(True, True), before.values(True, True)))
        rows = torch.tensor([3, 0])
        np.testing.assert_array_equal(
            env.render(env.goal_render_state(state[rows])).numpy(), env.render(goal).numpy()[rows.numpy()]
        )
        with pytest.raises((TypeError, ValueError)):
            env.set_state(goal)
    finally:
        env.close()
