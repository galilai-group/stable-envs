"""Render projections are rejected where a full simulator State is required."""

import pytest
import torch

import stable_envs as se
from tests.helpers.stepping import branch_step

NAVIGATION_KEYS = {
    "TwoRoom": {("agent", "position"), ("target", "position")},
    "Piecewise": {("agent", "position"), ("target", "position")},
    "SimplePointMaze": {
        ("agent", "position"),
        ("goal", "position"),
        ("task", "cached_goal_agent_position"),
        ("task", "state_is_float64"),
    },
}


@pytest.mark.parametrize(
    "name",
    [
        "CartPole",
        "Pendulum",
        *(pytest.param(name, marks=pytest.mark.family("navigation")) for name in NAVIGATION_KEYS),
        pytest.param("PushT", marks=pytest.mark.family("pusht")),
        pytest.param("PushTDiscrete", marks=pytest.mark.family("pusht")),
    ],
)
def test_lossy_projection_is_rejected_and_full_state_renders(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=4)
        if name in NAVIGATION_KEYS:
            assert set(state.keys(True, True)) == NAVIGATION_KEYS[name]
        if name in ("PushT", "PushTDiscrete"):
            rows = env.to_render_state(state)
            assert torch.equal(env.render(state), env.render(rows))
            incomplete = rows.clone()
        else:
            rows = env._pack_render_state(state)
            assert torch.equal(env.render(state), env._render(rows))
            incomplete = env._unpack_render_state(rows)
            with pytest.raises(ValueError, match="missing"):
                env.set_state(incomplete)
        for wrong in (rows, incomplete):
            with pytest.raises((TypeError, ValueError)):
                env.set_state(wrong)
            with pytest.raises((TypeError, ValueError)):
                branch_step(env, wrong, torch.zeros(2, env.action_dim))
        for attr in ("pack", "unpack", "pack_render_state", "unpack_render_state", "_state"):
            assert not hasattr(env, attr)
