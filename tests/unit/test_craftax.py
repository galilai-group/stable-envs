"""Craftax environments against Craftax called directly, environment masking, and State restored into another env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions, assert_instance, frame, upstream  # noqa: E402

pytestmark = pytest.mark.unit

SEED = 7
NUM_ENVS = 3


@pytest.mark.parametrize("name", NAMES)
def test_environments_match_craftax_called_directly(name):
    with se.make(name, num_envs=NUM_ENVS, device="cpu") as env:
        reset, step = upstream(env.reference_id)
        instances = [reset(jax.random.PRNGKey(SEED + instance)) for instance in range(NUM_ENVS)]
        state = env.reset(seed=SEED)
        frames = env.render(state)
        for instance, (obs, engine, key) in enumerate(instances):
            assert_instance(state, instance, engine, key)
            if name.endswith("Pixels"):
                np.testing.assert_array_equal(frames[instance].numpy(), frame(obs))
        for t, action in enumerate(actions(env, 12, SEED)):
            result = env.step(action)
            for instance, (_, engine, key) in enumerate(instances):
                obs, engine, reward, done, key = step(key, engine, int(action[instance, 0]))
                instances[instance] = (obs, engine, key)
                assert_instance(result["state"], instance, engine, key)
                assert result["reward"][instance].item() == float(reward), t
                assert result["terminated"][instance].item() == bool(done), t
                if name.endswith("Pixels"):
                    np.testing.assert_array_equal(result["frames"][instance, -1].numpy(), frame(obs), err_msg=str(t))
                else:
                    np.testing.assert_array_equal(result["observation"][instance].numpy(), np.asarray(obs))


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_an_inactive_environment_holds_its_state_and_reports_zeros(name):
    with se.make(name, num_envs=NUM_ENVS, device="cpu") as env:
        env.reset(seed=SEED)
        before = env.get_state()
        action = torch.as_tensor(actions(env, 1, SEED)[0])
        full = env._step(action)
        env.set_state(before)
        masked = env._step(action, active=torch.tensor([True, False, True]))
        assert_same_state(masked["state"][1], before[1])
        assert_same_state(env.get_state(), masked["state"])
        for instance in (0, 2):
            assert_same_state(masked[instance], full[instance])
        for key in masked.exclude("state").keys(True, True):
            assert not masked[key][1].any(), key


@pytest.mark.parametrize("name", NAMES)
def test_a_state_restores_into_another_env(name):
    with (
        se.make(name, num_envs=NUM_ENVS, device="cpu", seed=1) as env,
        se.make(name, num_envs=NUM_ENVS, device="cpu") as other,
    ):
        env.reset(seed=SEED)
        first, *later = actions(env, 4, SEED)
        env.step(first, frames=None)
        saved = env.get_state()
        other.reset(seed=SEED + 100)
        other.set_state(saved)
        assert_same_state(other.get_state(), saved)
        assert_same_state(other.render(), env.render(saved))
        for action in later:
            assert_same_state(other.step(action), env.step(action))
