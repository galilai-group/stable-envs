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


def _same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a, b) or (a.is_floating_point() and bool(torch.isclose(a, b, 0, 0, equal_nan=True).all()))


def _assert_equal(a, b):
    if torch.is_tensor(a):
        assert _same(a, b)
        return
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    assert [key for key in a.keys(True, True) if not _same(a[key], b[key])] == []


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
        _assert_equal(first.reset(seed=SEED), second.reset(seed=SEED))
        for action in _actions(first):
            _assert_equal(first.step(action, frames=None), second.step(action, frames=None))


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
        _assert_equal(got, expected)


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
        _assert_equal(saved, frozen)
        assert env.set_state(saved) is None
        _assert_equal(env.get_state(), saved)
        _assert_equal(env.step(action, frames=None), expected)
        for value in saved.values(True, True):
            value.zero_()
        _assert_equal(env.get_state(), expected["state"])


@pytest.mark.parametrize("name", envs("variation"))
def test_a_variation_round_trips_through_another_env(name):
    with _make(name) as first, _make(name) as second:
        if first.variation_dim == 0:
            return
        sampled = first.sample_variation(LANES, rng=np.random.default_rng(SEED))
        installed = first.set_variation(sampled).clone()
        torch.testing.assert_close(installed, torch.as_tensor(sampled, dtype=installed.dtype), rtol=0, atol=1e-6)
        _assert_equal(second.set_variation(installed), installed)
        state = first.reset(seed=SEED)
        _assert_equal(second.reset(seed=SEED), state)
        _assert_equal(second.get_variation(), first.get_variation())


@pytest.mark.parametrize("name", envs("batch_render"))
def test_a_lane_renders_the_same_alone_as_in_a_batch(name):
    with _make(name) as env:
        state = env.reset(seed=SEED)
        frames = env.render(state)
        for lane in range(LANES):
            _assert_equal(env.render(state[lane : lane + 1])[0], frames[lane])
