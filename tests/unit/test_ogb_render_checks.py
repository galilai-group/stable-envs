"""OGBench maze render(state) checks the drawn qpos fields; set_state checks the full State."""

import pytest

import stable_envs as se
from tests.helpers.render_checks import assert_render_checks_drawn_fields_only


@pytest.mark.parametrize("name", ["PointMazeMediumNavigate", "AntMazeMediumNavigate"])
def test_render_checks_drawn_fields_only(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        assert_render_checks_drawn_fields_only(env, env.reset(seed=5))
