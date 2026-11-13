import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.classic_control.simulation.local import CLASSIC_FIELDS
from tests.helpers.stepping import branch_step


@pytest.mark.parametrize(
    "domain,name,owned_field",
    [
        ("cartpole", "CartPole", "x"),
        ("pendulum", "Pendulum", "theta"),
        ("acrobot", "Acrobot", "theta1"),
        ("mountain_car", "MountainCar", "position"),
        ("mountain_car_continuous", "MountainCarContinuous", "position"),
    ],
)
def test_classic_state_is_complete_independent_and_renderable(domain, name, owned_field):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=17)
        assert env.execution_mode == ("DELEGATED_CPU" if env.simulation_backend == "gym" else "NATIVE_NUMPY")
        assert set(state.keys(True, True)) == set(CLASSIC_FIELDS[domain])
        for key, (width, dtype) in CLASSIC_FIELDS[domain].items():
            assert state[key].shape == (2, width)
            assert state[key].dtype == dtype
            assert env.state_spec[key].dtype == dtype

        held = state.clone()
        state[owned_field].add_(1)
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
        assert (env.get_state() == held).all()


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


@pytest.mark.parametrize("name", ["Acrobot", "MountainCar"])
def test_remaining_discrete_action_contract(name):
    with se.make(name, num_envs=1, device="cpu") as env:
        assert env.action_spec.dtype == torch.int64
        assert env.action_spec.discrete_n == 3
        out = env.step(np.ones((1, 1), np.int64), frames=None)
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
        with pytest.raises((AssertionError, TypeError)):
            env.step(np.ones((1, 1), np.float32), frames=None)


def test_continuous_mountain_car_dtype_transition_and_unclipped_action_cost():
    with se.make("MountainCarContinuous", num_envs=1, device="cpu") as env:
        reset = env.reset(seed=17)
        assert not reset["task", "state_is_float32"].item()
        precise = np.asarray([[0.1234567890123]], np.float64)
        out = env.step(precise, frames=None)
        assert out["state", "task", "state_is_float32"].item()
        saved = out["state"]
        expected = env.step(np.zeros((1, 1), np.float32), frames=None)
        env.set_state(saved)
        actual = env.step(np.zeros((1, 1), np.float32), frames=None)
        assert (actual == expected).all()

        env.reset(seed=17)
        clipped = env.step(np.asarray([[2.0]], np.float64), frames=None)
        assert clipped["reward"].item() == -0.4


def test_swm_float32_physics_defaults_are_applied_exactly():
    cases = {
        "CartPole": {"gravity": 9.8, "masspole": 0.1, "tau": 0.02},
        "Pendulum": {"dt": 0.05},
        "Acrobot": {"LINK_LENGTH_1": 1.0, "LINK_COM_POS_1": 0.5, "LINK_MOI": 1.0},
        "MountainCar": {"gravity": 0.0025, "force": 0.001, "max_speed": 0.07},
        "MountainCarContinuous": {"power": 0.0015, "max_speed": 0.07},
    }
    for name, expected in cases.items():
        with se.make(name, num_envs=1, device="cpu") as env:
            base = env._simulator.envs[0].unwrapped
            for attribute, value in expected.items():
                assert getattr(base, attribute) == float(np.float32(value)), (name, attribute)
            if name == "CartPole":
                assert base.total_mass == base.masspole + base.masscart
                assert base.polemass_length == base.masspole * base.length
