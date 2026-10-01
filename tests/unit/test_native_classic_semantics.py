"""Local contract checks for the two recovered classic-control defaults."""

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs._classic_semantics import CLASSIC_FIELDS
from tests.helpers.stepping import branch_step


@pytest.mark.parametrize("domain,name", [("cartpole", "CartPole"), ("pendulum", "Pendulum")])
def test_recovered_classic_state_is_complete_independent_and_renderable(domain, name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=17)
        assert env.execution_mode == "DELEGATED_CPU"
        assert set(state.keys(True, True)) == set(CLASSIC_FIELDS[domain])
        for key, (width, dtype) in CLASSIC_FIELDS[domain].items():
            assert state[key].shape == (2, width)
            assert state[key].dtype == dtype
            assert env.state_spec[key].dtype == dtype

        held = state.clone()
        state[next(iter(CLASSIC_FIELDS[domain]))].add_(1)
        assert (env.get_state() == held).all()
        assert torch.equal(env.render(held), env._render(env._pack_render_state(held)))

        compact = env._pack_render_state(held)
        with pytest.raises(TypeError, match="full State"):
            env.set_state(compact)
        with pytest.raises(ValueError, match="missing"):
            env.set_state(env._unpack_render_state(compact))
        assert not hasattr(env, "transition")
        assert not hasattr(env, "unroll")
        with pytest.raises(TypeError, match="full State"):
            branch_step(env, compact, torch.zeros((2, 1)))
        with pytest.raises(NotImplementedError, match="get_state"):
            env.sim_state()
        with pytest.raises(NotImplementedError, match="set_state"):
            env.set_sim_state(torch.zeros((2, len(env.sim_state_names))))


def test_cartpole_action_and_step_result_contract():
    with se.make("CartPole", num_envs=1, device="cpu") as env:
        assert env.action_spec.dtype == torch.int64
        assert env.action_spec.discrete_n == 2
        out = env.step(np.zeros((1, 1), np.int64), frames=None)
        assert set(out.keys()) == {
            "state",
            "reward",
            "terminated",
            "truncated",
            "observation",
            "info_state",
            "states",
            "valid",
            "state_valid",
        }
        assert out["reward"].item() == 1.0
        with pytest.raises(AssertionError):
            env.step(np.zeros((1, 1), np.float32), frames=None)


def test_pendulum_action_clipping_and_zero_reward_state():
    with se.make("Pendulum", num_envs=1, device="cpu") as env:
        assert env.action_spec.dtype == torch.float32
        state = env.get_state()
        state["theta"].zero_()
        state["theta_dot"].zero_()
        state["torque"].zero_()
        state["task", "has_last_u"].fill_(True)
        env.set_state(state)
        out = env.step(np.zeros((1, 1), np.float32), frames=None)
        assert out["reward"].item() == 0.0
        out = env.step(np.asarray([[3.25]], np.float64), frames=None)
        assert out["state", "torque"].item() == 2.0


def test_swm_float32_physics_defaults_are_applied_exactly():
    with se.make("CartPole", num_envs=1, device="cpu") as cart:
        base = cart._simulator.envs[0].unwrapped
        assert base.gravity == float(np.float32(9.8))
        assert base.masspole == float(np.float32(0.1))
        assert base.tau == float(np.float32(0.02))
        assert base.total_mass == base.masspole + base.masscart
        assert base.polemass_length == base.masspole * base.length
    with se.make("Pendulum", num_envs=1, device="cpu") as pendulum:
        base = pendulum._simulator.envs[0].unwrapped
        assert base.dt == float(np.float32(0.05))
