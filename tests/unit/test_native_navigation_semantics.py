import numpy as np
import pytest
import torch

import stable_envs as se


def test_simple_point_maze_preserves_float64_action_continuation():
    with se.make("SimplePointMaze", num_envs=1, device="cpu") as env:
        env.reset(seed=17)
        first = env.step(np.asarray([[0.1234567890123, -0.07]], np.float64), frames=None)
        assert se.get_field(first["state"], "agent.position").dtype == torch.float64
        assert se.get_field(first["state"], "task.state_is_float64").item()
        assert first["observation"].dtype == torch.float64
        saved = first["state"].clone()
        expected = env.step(np.asarray([[0.031234567, -0.017]], np.float32), frames=None)
        env.set_state(saved)
        actual = env.step(np.asarray([[0.031234567, -0.017]], np.float32), frames=None)
        assert torch.equal(
            se.get_field(actual["state"], "agent.position"), se.get_field(expected["state"], "agent.position")
        )


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise", "SimplePointMaze"])
def test_uncertified_sampled_reset_variations_refuse(name):
    with se.make(name, num_envs=1, device="cpu") as env:
        with pytest.raises(NotImplementedError, match="sampled"):
            env.reset(seed=0, options={"variation": ["background.color"]})


@pytest.mark.parametrize("name", ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "SimplePointMaze"])
def test_the_state_spec_keeps_bounds_and_samples_states_inside_them(name):
    with se.make(name, num_envs=3) as env:
        spec = env.state_spec
        rng = np.random.default_rng(0)
        for key, leaf in spec.items(nested=True):
            assert leaf.low is not None and leaf.high is not None, key
            assert leaf.contains(leaf.sample(5, rng)), key
        if name.startswith("TwoRoom"):
            assert spec["navigation", "float32"].names == ("agent_x", "agent_y", "target_x", "target_y")
