"""Craftax lanes against Craftax called directly, lane masking, and State restored into another env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions, assert_lane, frame, upstream  # noqa: E402

pytestmark = pytest.mark.unit

SEED = 7
LANES = 3


@pytest.mark.parametrize("name", NAMES)
def test_lanes_match_craftax_called_directly(name):
    with se.make(name, num_envs=LANES, device="cpu") as env:
        reset, step = upstream(env.reference_id)
        lanes = [reset(jax.random.PRNGKey(SEED + lane)) for lane in range(LANES)]
        state = env.reset(seed=SEED)
        frames = env.render(state)
        for lane, (obs, engine, key) in enumerate(lanes):
            assert_lane(state, lane, engine, key)
            if name.endswith("Pixels"):
                np.testing.assert_array_equal(frames[lane].numpy(), frame(obs))
        for t, action in enumerate(actions(env, 12, SEED)):
            result = env.step(action)
            for lane, (_, engine, key) in enumerate(lanes):
                obs, engine, reward, done, key = step(key, engine, int(action[lane, 0]))
                lanes[lane] = (obs, engine, key)
                assert_lane(result["state"], lane, engine, key)
                assert result["reward"][lane].item() == float(reward), t
                assert result["terminated"][lane].item() == bool(done), t
                if name.endswith("Pixels"):
                    np.testing.assert_array_equal(result["frames"][lane, -1].numpy(), frame(obs), err_msg=str(t))
                else:
                    np.testing.assert_array_equal(result["observation"][lane].numpy(), np.asarray(obs))


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_an_inactive_lane_holds_its_state_and_reports_zeros(name):
    with se.make(name, num_envs=LANES, device="cpu") as env:
        env.reset(seed=SEED)
        before = env.get_state()
        action = torch.as_tensor(actions(env, 1, SEED)[0])
        full = env._step(action)
        env.set_state(before)
        masked = env._step(action, active=torch.tensor([True, False, True]))
        assert_same_state(masked["state"][1], before[1])
        assert_same_state(env.get_state(), masked["state"])
        for lane in (0, 2):
            assert_same_state(masked[lane], full[lane])
        for key in masked.exclude("state").keys(True, True):
            assert not masked[key][1].any(), key


@pytest.mark.parametrize("name", NAMES)
def test_a_state_restores_into_another_env(name):
    with (
        se.make(name, num_envs=LANES, device="cpu", seed=1) as env,
        se.make(name, num_envs=LANES, device="cpu") as other,
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
