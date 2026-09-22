"""Shared recovery gate: projections must never impersonate simulator snapshots."""

import pytest
import torch
from tensordict import TensorDict

import stable_envs as se


@pytest.mark.parametrize("name", ["CartPole", "Pendulum", "Acrobot", "PushT", "MiniGridEmpty", "TwoRoom"])
def test_full_snapshot_restores_future_and_has_no_aliases(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        env.reset(seed=12)
        action = torch.ones(2, env.action_dim)
        if name == "PushT":
            action *= 250
        env.step(action)
        saved = env.get_state()
        assert isinstance(saved, TensorDict)
        for key in saved.keys(True, True):
            assert env.state_spec[key].dtype == saved[key].dtype
        frozen = saved.clone()
        expected = env.step(action)
        for _ in range(3):
            env.step(action)
        assert (saved == frozen).all()
        assert env.set_state(saved) is None
        assert (env.get_state() == saved).all()
        assert (env.step(action) == expected).all()
        for value in saved.values(True, True):
            value.zero_()
        assert (env.get_state() == expected["state"]).all()


@pytest.mark.parametrize("name", ["CartPole", "Pendulum", "PushT", "MiniGridEmpty"])
def test_lossy_projection_is_rejected_and_full_state_renders(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=4)
        rows = env._pack_render_state(state)
        assert torch.equal(env.render(state), env._render(rows))
        incomplete = env._unpack_render_state(rows)
        for wrong in (rows, incomplete):
            with pytest.raises((TypeError, ValueError)):
                env.set_state(wrong)
            with pytest.raises((TypeError, ValueError)):
                env.transition(wrong, torch.zeros(2, env.action_dim))
        for name in ("pack", "unpack", "pack_render_state", "unpack_render_state", "_state"):
            assert not hasattr(env, name)


def test_renderer_only_cell_does_not_claim_full_state():
    with se.make("CraftaxClassic", num_envs=1, device="cpu") as env:
        for call in (env.get_state, env.reset):
            with pytest.raises(NotImplementedError, match="full simulator semantics"):
                call()
