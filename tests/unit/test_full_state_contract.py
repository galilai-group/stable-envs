"""Shared recovery gate: projections must never impersonate simulator snapshots."""

import pytest
import torch

import stable_envs as se
from tests.helpers.stepping import branch_step


@pytest.mark.parametrize(
    "name",
    [
        "CartPole",
        "Pendulum",
        pytest.param("PushT", marks=pytest.mark.family("pusht")),
        pytest.param("PushTDiscrete", marks=pytest.mark.family("pusht")),
    ],
)
def test_lossy_projection_is_rejected_and_full_state_renders(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=4)
        if name in ("PushT", "PushTDiscrete"):
            rows = env.to_render_state(state)
            assert torch.equal(env.render(state), env.render(rows))
            incomplete = rows.clone()
        else:
            rows = env._pack_render_state(state)
            assert torch.equal(env.render(state), env._render(rows))
            incomplete = env._unpack_render_state(rows)
        for wrong in (rows, incomplete):
            with pytest.raises((TypeError, ValueError)):
                env.set_state(wrong)
            with pytest.raises((TypeError, ValueError)):
                branch_step(env, wrong, torch.zeros(2, env.action_dim))
        for attr in ("pack", "unpack", "pack_render_state", "unpack_render_state", "_state"):
            assert not hasattr(env, attr)


def test_renderer_only_cell_does_not_claim_full_state():
    with se.make("CraftaxClassic", num_envs=1, device="cpu") as env:
        for call in (env.get_state, env.reset):
            with pytest.raises(NotImplementedError, match="full simulator semantics"):
                call()
