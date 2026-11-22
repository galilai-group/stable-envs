"""``Env.info`` of a class that does not compute info."""

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


def test_base_info_raises_not_implemented_with_the_class_name():
    env = Rows(num_envs=2, device="cpu")
    env._set_render_state(torch.zeros(2, 2))
    with pytest.raises(NotImplementedError, match=r"Rows\.info"):
        env.info(env._unpack_render_state(torch.zeros(2, 2)))


def test_info_of_a_closed_env_raises_runtime_error():
    env = Rows(num_envs=1, device="cpu")
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.info(None)
