import numpy as np
import pytest
import torch

import stable_envs as se


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("TwoRoom", {("agent", "position"), ("target", "position")}),
        ("Piecewise", {("agent", "position"), ("target", "position")}),
        (
            "SimplePointMaze",
            {
                ("agent", "position"),
                ("goal", "position"),
                ("task", "cached_goal_agent_position"),
                ("task", "state_is_float64"),
            },
        ),
    ],
)
def test_full_state_rejects_lossy_render_projection(name, keys):
    with se.make(name, num_envs=1, device="cpu") as env:
        state = env.reset(seed=17)
        assert set(state.keys(True, True)) == keys
        compact = env._unpack_render_state(env._pack_render_state(state))
        with pytest.raises(ValueError, match="missing"):
            env.set_state(compact)


def test_simple_point_maze_preserves_float64_action_continuation():
    with se.make("SimplePointMaze", num_envs=1, device="cpu") as env:
        env.reset(seed=17)
        first = env.step(np.asarray([[0.1234567890123, -0.07]], np.float64), frames=None)
        assert first["state", "agent", "position"].dtype == torch.float64
        assert first["state", "task", "state_is_float64"].item()
        assert first["observation"].dtype == torch.float64
        saved = first["state"].clone()
        expected = env.step(np.asarray([[0.031234567, -0.017]], np.float32), frames=None)
        env.set_state(saved)
        actual = env.step(np.asarray([[0.031234567, -0.017]], np.float32), frames=None)
        assert torch.equal(actual["state", "agent", "position"], expected["state", "agent", "position"])


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise", "SimplePointMaze"])
def test_uncertified_sampled_reset_variations_refuse(name):
    with se.make(name, num_envs=1, device="cpu") as env:
        with pytest.raises(NotImplementedError, match="sampled"):
            env.reset(seed=0, options={"variation": ["background.color"]})
