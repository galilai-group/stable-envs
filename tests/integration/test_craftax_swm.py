"""Craftax environments against stable-worldmodel's CraftaxWrapper, one wrapper per environment."""

from __future__ import annotations

import numpy as np
import pytest

import stable_envs as se

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions, assert_instance  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.reference, pytest.mark.swm]

SEED = 3
NUM_ENVS = 3


@pytest.mark.parametrize("name", NAMES)
def test_environments_match_stable_worldmodel(name):
    pytest.importorskip("stable_worldmodel")
    from stable_worldmodel.envs.craftax import craftax as swm

    wrappers = [
        getattr(swm, f"{name}Wrapper")(seed=SEED + instance, render_mode="rgb_array") for instance in range(NUM_ENVS)
    ]
    with se.make(name, num_envs=NUM_ENVS, device="cpu", seed=SEED) as env:
        for w in wrappers:
            w.reset()
        state = env.reset()
        for action in actions(env, 12, SEED):
            for instance, w in enumerate(wrappers):
                assert_instance(state, instance, w._state, w._key)
                if name.endswith("Pixels"):
                    np.testing.assert_array_equal(env.render(state[instance : instance + 1])[0].numpy(), w.render())
            result = env.step(action, frames=None)
            state = result["state"]
            for instance, w in enumerate(wrappers):
                obs, reward, terminated, truncated, _ = w.step(action[instance, 0])
                assert (result["reward"][instance].item(), result["terminated"][instance].item()) == (
                    reward,
                    terminated,
                )
                assert truncated is False and not result["truncated"][instance]
                if not name.endswith("Pixels"):
                    np.testing.assert_array_equal(result["observation"][instance].numpy(), obs)
        for instance, w in enumerate(wrappers):
            assert_instance(state, instance, w._state, w._key)


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_reset_without_a_seed_continues_the_keys_as_stable_worldmodel_does(name):
    pytest.importorskip("stable_worldmodel")
    from stable_worldmodel.envs.craftax import craftax as swm

    wrappers = [getattr(swm, f"{name}Wrapper")(seed=0, render_mode="rgb_array") for _ in range(NUM_ENVS)]
    with se.make(name, num_envs=NUM_ENVS, device="cpu", seed=SEED + 50) as env:
        env.reset(seed=SEED)
        for instance, w in enumerate(wrappers):
            w.reset(seed=SEED + instance)
        first, second = actions(env, 2, SEED)
        env.step(first, frames=None)
        for instance, w in enumerate(wrappers):
            w.step(first[instance, 0])
        state = env.reset()
        for instance, w in enumerate(wrappers):
            w.reset()
            assert_instance(state, instance, w._state, w._key)
        state = env.step(second, frames=None)["state"]
        for instance, w in enumerate(wrappers):
            w.step(second[instance, 0])
            assert_instance(state, instance, w._state, w._key)
