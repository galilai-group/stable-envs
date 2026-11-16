"""Craftax lanes against stable-worldmodel's CraftaxWrapper, one wrapper per lane."""

from __future__ import annotations

import numpy as np
import pytest

import stable_envs as se

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions, assert_lane  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.reference, pytest.mark.swm]

SEED = 3
LANES = 3


@pytest.mark.parametrize("name", NAMES)
def test_lanes_match_stable_worldmodel(name):
    pytest.importorskip("stable_worldmodel")
    from stable_worldmodel.envs.craftax import craftax as swm

    wrappers = [getattr(swm, f"{name}Wrapper")(seed=SEED + lane, render_mode="rgb_array") for lane in range(LANES)]
    with se.make(name, num_envs=LANES, device="cpu", seed=SEED) as env:
        for w in wrappers:
            w.reset()
        state = env.reset()
        for action in actions(env, 12, SEED):
            for lane, w in enumerate(wrappers):
                assert_lane(state, lane, w._state, w._key)
                if name.endswith("Pixels"):
                    np.testing.assert_array_equal(env.render(state[lane : lane + 1])[0].numpy(), w.render())
            result = env.step(action, frames=None)
            state = result["state"]
            for lane, w in enumerate(wrappers):
                obs, reward, terminated, truncated, _ = w.step(action[lane, 0])
                assert (result["reward"][lane].item(), result["terminated"][lane].item()) == (reward, terminated)
                assert truncated is False and not result["truncated"][lane]
                if not name.endswith("Pixels"):
                    np.testing.assert_array_equal(result["observation"][lane].numpy(), obs)
        for lane, w in enumerate(wrappers):
            assert_lane(state, lane, w._state, w._key)


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_reset_without_a_seed_continues_the_keys_as_stable_worldmodel_does(name):
    pytest.importorskip("stable_worldmodel")
    from stable_worldmodel.envs.craftax import craftax as swm

    wrappers = [getattr(swm, f"{name}Wrapper")(seed=0, render_mode="rgb_array") for _ in range(LANES)]
    with se.make(name, num_envs=LANES, device="cpu", seed=SEED + 50) as env:
        env.reset(seed=SEED)
        for lane, w in enumerate(wrappers):
            w.reset(seed=SEED + lane)
        first, second = actions(env, 2, SEED)
        env.step(first, frames=None)
        for lane, w in enumerate(wrappers):
            w.step(first[lane, 0])
        state = env.reset()
        for lane, w in enumerate(wrappers):
            w.reset()
            assert_lane(state, lane, w._state, w._key)
        state = env.step(second, frames=None)["state"]
        for lane, w in enumerate(wrappers):
            w.step(second[lane, 0])
            assert_lane(state, lane, w._state, w._key)
