"""The generic contract, run on every registered env.

Each case belongs to its env's family (``ci/families.toml``), so it runs in that family's CI profile
and is skipped where the family is not installed. Waivers live only in ``tests/exemptions.toml``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.exemptions import param
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

LANES = 3
SEED = 11
STEPS = 3


def envs(check):
    return [param(name, se.registry.REGISTRY[name], check) for name in se.list_envs()]


def _make(name, num_envs=LANES):
    return se.make(name, num_envs=num_envs, device="cpu")


def _actions(env, seed=SEED, steps=STEPS):
    rng = np.random.default_rng(seed)
    return [env.action_spec.sample(env.num_envs, rng) for _ in range(steps)]


def _assert_state_fits(env, state):
    assert state.batch_size == torch.Size([env.num_envs])
    env.state_spec.validate(state, owner="state")
    for key, spec in env.state_spec.items(nested=True):
        assert state[key].dtype == spec.dtype, key


@pytest.mark.parametrize("name", envs("specs"))
def test_reset_step_and_render_match_their_specs(name):
    with _make(name) as env:
        _assert_state_fits(env, env.reset(seed=SEED))
        action = _actions(env, steps=1)[0]
        assert env.action_spec.contains(action)
        result = env.step(action)
        _assert_state_fits(env, result["state"])
        frame = tuple(env.frame_spec.shape)
        assert result["frames"].dtype == torch.uint8 and tuple(result["frames"].shape) == (env.num_envs, 1, *frame)
        rendered = env.render(result["state"])
        assert rendered.dtype == torch.uint8 and tuple(rendered.shape) == (env.num_envs, *frame)
        assert torch.equal(rendered, result["frames"][:, -1])


@pytest.mark.parametrize("name", envs("seeded"))
def test_the_same_seed_gives_the_same_rollout(name):
    with _make(name) as first, _make(name) as second:
        assert_same_state(first.reset(seed=SEED), second.reset(seed=SEED))
        for action in _actions(first):
            assert_same_state(first.step(action, frames=None), second.step(action, frames=None))


@pytest.mark.parametrize("name", envs("lanes"))
def test_a_lane_does_not_see_its_neighbours_actions(name):
    with _make(name) as env:
        env.reset(seed=SEED)
        start = env.get_state()
        actions = _actions(env)
        for action in actions:
            expected = env.step(action, frames=None)["state"][0]
        env.set_state(start)
        others = _actions(env, seed=SEED + 1)
        for action, other in zip(actions, others):
            other[0] = action[0]
            got = env.step(other, frames=None)["state"][0]
        assert_same_state(got, expected)


@pytest.mark.parametrize("name", envs("state"))
def test_a_saved_state_restores_bit_exactly_and_owns_its_memory(name):
    with _make(name) as env:
        env.reset(seed=SEED)
        first, action, *later = _actions(env)
        env.step(first, frames=None)
        saved = env.get_state()
        frozen = saved.clone()
        expected = env.step(action, frames=None)
        for other in later:
            env.step(other, frames=None)
        assert_same_state(saved, frozen)
        assert env.set_state(saved) is None
        assert_same_state(env.get_state(), saved)
        assert_same_state(env.step(action, frames=None), expected)
        for value in saved.values(True, True):
            value.zero_()
        assert_same_state(env.get_state(), expected["state"])


@pytest.mark.parametrize("name", envs("variation"))
def test_a_variation_round_trips_through_another_env(name):
    with _make(name) as first, _make(name) as second:
        if first.variation_dim == 0:
            return
        sampled = first.sample_variation(LANES, rng=np.random.default_rng(SEED))
        installed = first.set_variation(sampled).clone()
        torch.testing.assert_close(installed, torch.as_tensor(sampled, dtype=installed.dtype), rtol=0, atol=1e-6)
        assert_same_state(second.set_variation(installed), installed)
        state = first.reset(seed=SEED)
        assert_same_state(second.reset(seed=SEED), state)
        assert_same_state(second.get_variation(), first.get_variation())


@pytest.mark.parametrize("name", envs("batch_render"))
def test_a_lane_renders_the_same_alone_as_in_a_batch(name):
    with _make(name) as env:
        state = env.reset(seed=SEED)
        frames = env.render(state)
        for lane in range(LANES):
            assert_same_state(env.render(state[lane : lane + 1])[0], frames[lane])


@pytest.mark.parametrize("name", envs("native_size"))
def test_a_class_draws_any_size_unless_it_declares_its_native_size(name):
    cls = se.registry.REGISTRY[name]
    if cls.native_size is None:
        with se.make(name, num_envs=1, size=128, device="cpu") as env:
            assert tuple(env.render(env.reset(seed=SEED)).shape[1:3]) == (128, 128)
        return
    assert cls.default_size == cls.native_size
    with _make(name) as env:
        assert env.size == cls.native_size
        assert tuple(env.render(env.reset(seed=SEED)).shape[1:]) == tuple(env.frame_spec.shape)
    with pytest.raises(ValueError):
        with se.make(name, num_envs=1, size=cls.native_size + 8, device="cpu") as env:
            env.render(env.reset(seed=SEED))
