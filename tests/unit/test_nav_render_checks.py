"""render(state) checks the navigation fields the renderer draws, including Piecewise's hidden target."""

import pytest
import torch

import stable_envs as se
from stable_envs._core.state_schema import unfold


def test_hidden_target_is_checked_only_where_it_is_drawn():
    with se.make("TwoRoom", num_envs=2, device="cpu") as env:
        state = env.reset(seed=1)
        frames = env.render(state)
        bad = state.clone()
        se.get_field(bad, "target.position")[0, 0] = float("nan")
        assert torch.equal(env.render(bad), frames)
        with pytest.raises(ValueError, match="target.position contains non-finite"):
            env.set_state(bad)
        bad = state.clone()
        se.get_field(bad, "agent.position")[1, 1] = float("nan")
        with pytest.raises(ValueError, match="agent.position contains non-finite"):
            env.render(bad)

    with se.make("Piecewise", num_envs=2, device="cpu") as env:
        state = env.reset(seed=1)
        bad = state.clone()
        se.get_field(bad, "target.position")[0, 0] = float("nan")
        with pytest.raises(ValueError, match="target.position contains non-finite"):
            env.render(bad)
        with pytest.raises(ValueError, match="missing target.position"):
            env.render(unfold(state).exclude(("target", "position")))
        with pytest.raises(ValueError, match="must be torch.float32"):
            env.render(state.apply(lambda group: group[..., :3]))


def test_simple_point_maze_render_skips_restore_only_fields():
    with se.make("SimplePointMaze", num_envs=2, device="cpu") as env:
        state = env.reset(seed=2)
        frames = env.render(state)
        bad = state.clone()
        se.get_field(bad, "task.cached_goal_agent_position")[1] = float("nan")
        assert torch.equal(env.render(bad), frames)
        with pytest.raises(ValueError, match="cached_goal_agent_position contains non-finite"):
            env.set_state(bad)
        bad = state.clone()
        se.get_field(bad, "goal.position")[0, 0] = float("inf")
        with pytest.raises(ValueError, match="goal.position contains non-finite"):
            env.render(bad)
