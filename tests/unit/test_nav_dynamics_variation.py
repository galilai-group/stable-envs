"""Native navigation Variation leaves that configure dynamics."""

import pytest
import torch

import stable_envs as snv

pytestmark = pytest.mark.unit


def test_two_room_speed_scales_the_step():
    env = snv.TwoRoom(num_envs=1, device="cpu")
    state = env.reset(seed=0)
    action = torch.tensor([[1.0, 0.0]])
    env.set_variation({"agent.speed": [1.0]})
    slow = env.transition(state, action)
    env.set_variation({"agent.speed": [5.0]})
    fast = env.transition(state, action)
    dx_slow = float((slow["agent", "position"] - state["agent", "position"])[0, 0])
    dx_fast = float((fast["agent", "position"] - state["agent", "position"])[0, 0])
    assert dx_fast > dx_slow > 0
    env.close()


def test_piecewise_zone_bias_is_a_variation_leaf():
    env = snv.Piecewise(num_envs=1, device="cpu")
    assert "zones.bias_0" in env.variation_space.names()
    assert "agent.speed" in env.dynamics_variation_leaves
    env.close()
