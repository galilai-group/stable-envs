"""render(state) checks the fields the classic-control renderer draws; set_state checks the full State."""

import pytest
import torch

import stable_envs as se
from stable_envs import get_field


@pytest.mark.parametrize("name,drawn,hidden", [("CartPole", "theta", "x_dot"), ("Pendulum", "torque", "theta_dot")])
def test_render_checks_only_drawn_fields(name, drawn, hidden):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=3)
        frames = env.render(state)

        restore_only = state.clone()
        get_field(restore_only, hidden)[1, 0] = float("nan")
        get_field(restore_only, "lifecycle/elapsed_steps")[0, 0] = -1
        get_field(restore_only, "rng/state")[0, 0] = -5
        assert torch.equal(env.render(restore_only), frames)
        with pytest.raises(ValueError, match=f"{hidden}.*non-finite"):
            env.set_state(restore_only)

        bad = state.clone()
        get_field(bad, drawn)[1, 0] = float("inf")
        with pytest.raises(ValueError, match=f"{drawn} contains non-finite"):
            env.render(bad)
        group = env._group
        bad = state.clone()
        bad[group] = bad[group].to(torch.float32)
        with pytest.raises(ValueError, match="float64.*got.*float32"):
            env.render(bad)
        bad = state.clone()
        bad[group] = torch.cat([bad[group], bad[group]], 1)
        with pytest.raises(ValueError, match="group .* must be"):
            env.render(bad)
        with pytest.raises(ValueError, match="missing the group"):
            env.render(state.exclude(group))
        assert torch.equal(env.render(env._pack_render_state(state)), frames)
