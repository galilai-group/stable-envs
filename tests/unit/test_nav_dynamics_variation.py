"""Native navigation Variation leaves that configure dynamics."""

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs.envs.navigation.envs import ZONE_BIAS
from tests.helpers.stepping import branch_step

pytestmark = pytest.mark.unit


def test_two_room_speed_scales_the_step():
    env = snv.TwoRoom(num_envs=1, device="cpu")
    state = env.reset(seed=0)
    action = torch.tensor([[1.0, 0.0]])
    env.set_variation({"agent.speed": [1.0]})
    slow = branch_step(env, state, action)
    env.set_variation({"agent.speed": [5.0]})
    fast = branch_step(env, state, action)
    dx_slow = float((slow["agent", "position"] - state["agent", "position"])[0, 0])
    dx_fast = float((fast["agent", "position"] - state["agent", "position"])[0, 0])
    assert dx_fast > dx_slow > 0
    env.close()


def test_piecewise_zone_bias_shifts_only_the_step_taken_in_that_zone():
    env = snv.Piecewise(num_envs=1, device="cpu")
    state = env.reset(seed=0)
    idle = torch.zeros(1, 2)
    default = branch_step(env, state, idle)["agent", "position"]
    shifts = []
    for zone in range(4):
        env.set_variation({f"zones.bias_{zone}": [[7.0, 0.0]]})
        shifts.append(branch_step(env, state, idle)["agent", "position"] - default)
        env.set_variation({f"zones.bias_{zone}": [list(ZONE_BIAS[zone])]})
    moved = [i for i, shift in enumerate(shifts) if shift.abs().max() > 0]
    assert len(moved) == 1
    expected = torch.tensor([[7.0, 0.0]]) - torch.tensor([ZONE_BIAS[moved[0]]])
    assert torch.allclose(shifts[moved[0]], expected)
    env.close()


def test_style_leaves_follow_the_rule():
    with snv.make("TwoRoom", num_envs=2, device="cpu") as env:
        registered = float(env._style["agent.speed"][0])
        env.reset(seed=1, options={"variation_values": {"agent.speed": [6.5]}})
        assert (env._style["agent.speed"] == 6.5).all()
        env.reset(seed=1)
        assert (env._style["agent.speed"] == registered).all()

        env.set_variation({"agent.speed": [4.0], "agent.color": [10, 20, 30]})
        env.reset(seed=2, options={"variation_values": {"agent.speed": [[6.5], [7.0]]}})
        assert env._style["agent.speed"].tolist() == [6.5, 7.0]
        assert env._style["agent.color"].tolist() == [[10.0, 20.0, 30.0]] * 2
        env.reset(seed=2)
        assert (env._style["agent.speed"] == 4.0).all()
        assert env._style["agent.color"].tolist() == [[10.0, 20.0, 30.0]] * 2


def test_reset_leaves_take_one_value_per_lane_and_follow_the_rule():
    starts = np.asarray([[31.25, 47.5], [180.0, 60.0]], np.float32)
    with snv.make("TwoRoom", num_envs=2, device="cpu") as env:
        assert [leaf.path for leaf in env.reset_leaves] == ["agent.position", "target.position"]
        sampled = env.reset(seed=3)["agent", "position"]
        state = env.reset(seed=3, options={"variation_values": {"agent.position": starts}})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), starts)
        assert torch.equal(env.reset(seed=3)["agent", "position"], sampled)
        state = env.reset(seed=3, options={"variation": []})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), [[60.0, 112.0]] * 2)
        np.testing.assert_array_equal(state["target", "position"].numpy(), [[164.0, 112.0]] * 2)
        with pytest.raises(ValueError, match="per lane"):
            env.reset(options={"variation_values": {"agent.position": np.zeros((3, 2))}})
    with snv.make("SimplePointMaze", num_envs=2, device="cpu") as env:
        goals = env.reset(seed=4, options={"variation": ["goal.position"]})["goal", "position"].numpy()
        assert len({tuple(row) for row in goals.tolist()} - {(4.5, 4.5)}) == 2
        state = env.reset(seed=4, options={"variation": [], "variation_values": {"goal.position": [2.5, 1.5]}})
        np.testing.assert_array_equal(state["goal", "position"].numpy(), [[2.5, 1.5]] * 2)
