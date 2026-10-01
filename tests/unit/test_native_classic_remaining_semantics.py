"""Local contract checks for the remaining recovered classic-control defaults."""

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs._classic_semantics import CLASSIC_FIELDS
from tests.helpers.stepping import branch_step

CASES = (
    ("acrobot", "Acrobot"),
    ("mountain_car", "MountainCar"),
    ("mountain_car_continuous", "MountainCarContinuous"),
)


@pytest.mark.parametrize("domain,name", CASES)
def test_remaining_classic_state_is_complete_independent_and_renderable(domain, name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=17)
        assert env.execution_mode == "DELEGATED_CPU"
        assert set(state.keys(True, True)) == set(CLASSIC_FIELDS[domain])
        for key, (width, dtype) in CLASSIC_FIELDS[domain].items():
            assert state[key].shape == (2, width)
            assert state[key].dtype == dtype
            assert env.state_spec[key].dtype == dtype

        held = state.clone()
        state[next(key for key in CLASSIC_FIELDS[domain] if isinstance(key, str))].add_(1)
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


def test_remaining_swm_float32_physics_defaults_are_applied_exactly():
    cases = {
        "Acrobot": {"LINK_LENGTH_1": 1.0, "LINK_COM_POS_1": 0.5, "LINK_MOI": 1.0},
        "MountainCar": {"gravity": 0.0025, "force": 0.001, "max_speed": 0.07},
        "MountainCarContinuous": {"power": 0.0015, "max_speed": 0.07},
    }
    for name, expected in cases.items():
        with se.make(name, num_envs=1, device="cpu") as env:
            base = env._simulator.envs[0].unwrapped
            for attribute, value in expected.items():
                assert getattr(base, attribute) == float(np.float32(value))
