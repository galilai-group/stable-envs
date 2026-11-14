"""Craftax variation leaves set upstream EnvParams per lane, and the State keeps them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import actions, assert_lane  # noqa: E402

pytestmark = pytest.mark.unit

SEED = 7
LANES = 3
VALUES = {
    "CraftaxClassicSymbolic": {
        "world.day_length": [300, 60, 900],
        "world.always_diamond": [1, 0, 1],
        "zombie.health": [5, 1, 9],
        "cow.health": [3, 7, 1],
        "skeleton.health": [3, 2, 8],
        "mobs.despawn_distance": [14, 6, 30],
        "spawn.cow": [0.1, 1.0, 0.0],
        "spawn.zombie_base": [0.02, 0.9, 0.0],
        "spawn.zombie_night": [0.1, 0.5, 0.0],
        "spawn.skeleton": [0.05, 1.0, 0.0],
    },
    "CraftaxPixels": {
        "world.day_length": [300, 60, 900],
        "world.always_diamond": [0, 1, 1],
        "mobs.despawn_distance": [14, 6, 30],
        "player.max_attribute": [5, 1, 10],
    },
}


def _upstream(env, lane: int, values: dict):
    """Single-lane upstream reset and step with the wrapper key handling, given these EnvParams values.

    The values are arguments of the jitted calls, as when a caller varies EnvParams upstream.
    """
    from craftax.craftax_env import make_craftax_env_from_name

    upstream = make_craftax_env_from_name(env.reference_id, auto_reset=False)
    cast = {bool: bool, int: int, float: np.float32}
    params = upstream.default_params.replace(
        **{attr: cast[type(default)](values[path][lane]) for path, attr, _, default in env._k.params if path in values}
    )
    params = jax.tree.map(jax.numpy.asarray, params)

    def reset(key, params):
        kept, used = jax.random.split(key)
        return upstream.reset(used, params)[1], kept

    def step(key, state, action, params):
        kept, used = jax.random.split(key)
        _, state, reward, done, _ = upstream.step(used, state, action, params)
        return state, reward, done, kept

    reset, step = jax.jit(reset), jax.jit(step)
    return lambda key: reset(key, params), lambda key, state, action: step(key, state, action, params)


@pytest.mark.parametrize("name", sorted(VALUES))
def test_per_lane_values_step_as_upstream_steps_with_those_params(name):
    values = VALUES[name]
    with se.make(name, num_envs=LANES, device="cpu") as env, jax.default_device(jax.devices("cpu")[0]):
        state = env.reset(seed=SEED, options={"variation_values": values})
        lanes = []
        for lane in range(LANES):
            reset, step = _upstream(env, lane, values)
            engine, key = reset(jax.random.PRNGKey(SEED + lane))
            assert_lane(state, lane, engine, key)
            lanes.append([engine, key, step])
        for path, column in values.items():
            stored = state[("physics", *path.split("."))].reshape(-1).double().numpy()
            np.testing.assert_allclose(stored, np.asarray(column, np.float32), rtol=0, atol=0, err_msg=path)
        for action in actions(env, 12, SEED):
            result = env.step(action, frames=None)
            for lane, (engine, key, step) in enumerate(lanes):
                engine, reward, done, key = step(key, engine, int(action[lane, 0]))
                lanes[lane][:2] = [engine, key]
                assert_lane(result["state"], lane, engine, key)
                assert result["reward"][lane].item() == float(reward)
                assert result["terminated"][lane].item() == bool(done)


@pytest.mark.parametrize("name", sorted(VALUES))
def test_a_state_carries_its_values_into_another_env(name):
    with (
        se.make(name, num_envs=LANES, device="cpu") as env,
        se.make(name, num_envs=LANES, device="cpu") as other,
    ):
        env.reset(seed=SEED, options={"variation_values": VALUES[name]})
        first, *later = actions(env, 4, SEED)
        env.step(first, frames=None)
        saved = env.get_state()
        other.reset(seed=SEED + 1)
        other.set_state(saved)
        assert torch.equal(other.get_variation(), env.get_variation())
        for action in later:
            assert_same_state(other.step(action, frames=None), env.step(action, frames=None))
        # The next reset without options returns to the installed defaults.
        state = env.reset(seed=SEED)
        with se.make(name, num_envs=LANES, device="cpu") as fresh:
            assert_same_state(state, fresh.reset(seed=SEED))


def test_values_outside_a_leaf_are_refused_by_set_state():
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cpu") as env:
        state = env.reset(seed=SEED)
        before = env.get_state()
        state["physics", "zombie", "health"][1] = 0
        with pytest.raises(ValueError, match="physics.zombie.health"):
            env.set_state(state)
        assert_same_state(env.get_state(), before)


def test_sampled_values_stay_inside_their_leaves():
    with se.make("CraftaxClassicSymbolic", num_envs=8, device="cpu") as env:
        paths = sorted(env.dynamics_variation_leaves)
        state = env.reset(seed=SEED, options={"variation": paths})
        for path in paths:
            space = env.variation_space._walk(path)
            value = state[("physics", *path.split("."))].reshape(-1).double().numpy()
            low, high = (space.start, space.start + space.n - 1) if hasattr(space, "n") else (space.low, space.high)
            assert ((value >= low) & (value <= high)).all(), path
        rows = env.sample_variation(64)
        for path, offset, _ in env.variation_space.flat_layout():
            space = env.variation_space._walk(path)
            if hasattr(space, "n"):
                assert rows[:, offset].min() >= space.start and rows[:, offset].max() < space.start + space.n, path


def test_render_fills_out_and_checks_variation():
    with se.make("CraftaxClassicPixels", num_envs=LANES, device="cpu") as env:
        state = env.reset(seed=SEED)
        frames = env.render(state)
        out = torch.zeros_like(frames)
        assert env.render(state, out=out) is out and torch.equal(out, frames)
        live = torch.zeros_like(frames)
        env.render(out=live)
        assert torch.equal(live, frames)
        assert torch.equal(env.render(state, variation={"world.day_length": 60}), frames)
        with pytest.raises(KeyError):
            env.render(state, variation={"no.such_leaf": 1})
        with pytest.raises(ValueError):
            env.render(state, out=torch.zeros((LANES, 2, 2, 3), dtype=torch.uint8))
