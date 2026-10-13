"""Native navigation Variation leaves that configure dynamics."""

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
