"""``Env.goal_render_state`` of a class without a goal configuration."""

from __future__ import annotations

import pytest
import torch

from stable_envs.env import Env
from stable_envs.spaces import Dict

pytestmark = pytest.mark.unit


class Rows(Env):
    state_dim = 2
    state_names = ("x", "y")
    state_layout = ((("position",), 2, 0),)

    def make_variation_space(self):
        return Dict({})


def test_base_goal_render_state_raises_not_implemented_with_the_class_name():
    env = Rows(num_envs=2, device="cpu")
    with pytest.raises(NotImplementedError, match=r"Rows\.goal_render_state: this class has no goal configuration"):
        env.goal_render_state(env._unpack_render_state(torch.zeros(2, 2)))


def test_goal_render_state_of_a_closed_env_raises_runtime_error():
    env = Rows(num_envs=1, device="cpu")
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.goal_render_state(None)
