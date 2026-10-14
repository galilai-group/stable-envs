"""Craftax driven one lane at a time, and the comparison of one State lane with upstream state."""

from __future__ import annotations

import jax
import numpy as np

NAMES = ("CraftaxClassicPixels", "CraftaxClassicSymbolic", "CraftaxPixels", "CraftaxSymbolic")


def frame(obs) -> np.ndarray:
    """A pixel observation as stable-worldmodel's CraftaxWrapper renders it."""
    return (np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def actions(env, steps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, env.action_high + 1, size=(steps, env.num_envs, 1))


def upstream(reference_id: str):
    """Jitted single-lane reset and step with the wrapper key handling of stable-worldmodel's CraftaxWrapper."""
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(reference_id, auto_reset=False)
    params = env.default_params

    def reset(key):
        kept, used = jax.random.split(key)
        obs, state = env.reset(used, params)
        return obs, state, kept

    def step(key, state, action):
        kept, used = jax.random.split(key)
        obs, state, reward, done, _ = env.step(used, state, action, params)
        return obs, state, reward, done, kept

    return jax.jit(reset), jax.jit(step)


def assert_lane(state, lane: int, engine, key) -> None:
    """Lane ``lane`` of a State equals an upstream ``EnvState`` and wrapper key, leaf for leaf."""
    for path, leaf in [*jax.tree_util.tree_flatten_with_path(engine)[0], ((), key)]:
        name = ("engine", *(entry.name for entry in path)) if path else ("key",)
        expected = np.asarray(leaf).reshape(-1)
        if expected.dtype == np.uint32:
            expected = expected.view(np.int32)
        np.testing.assert_array_equal(state[name][lane].cpu().numpy(), expected, err_msg=".".join(name))
