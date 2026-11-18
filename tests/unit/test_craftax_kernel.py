"""Craftax episode ends, the step limit, seed keys, render batching and the device choice."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs import get_field

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import actions, assert_instance, upstream  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

pytestmark = pytest.mark.unit

SEED = 7
NUM_ENVS = 3


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_an_environment_that_dies_and_an_environment_at_the_step_limit_end_as_upstream_ends_them(name):
    from craftax.craftax_env import make_craftax_env_from_name

    with se.make(name, num_envs=NUM_ENVS, device="cpu") as env, jax.default_device(jax.devices("cpu")[0]):
        limit = int(make_craftax_env_from_name(env.reference_id, auto_reset=False).default_params.max_timesteps)
        reset, step = upstream(env.reference_id)
        instances = [list(reset(jax.random.PRNGKey(SEED + instance))[1:]) for instance in range(NUM_ENVS)]
        state = env.reset(seed=SEED)
        get_field(state, "engine/player_health")[1] = 0
        get_field(state, "engine/timestep")[2] = limit - 1
        instances[1][0] = instances[1][0].replace(player_health=instances[1][0].player_health * 0)
        instances[2][0] = instances[2][0].replace(timestep=instances[2][0].timestep * 0 + limit - 1)
        env.set_state(state)
        sequence = actions(env, 3, SEED)
        out = env.step(torch.as_tensor(sequence).permute(1, 0, 2), states=slice(None), frames=None)
        assert out["valid"].tolist() == [[True] * 3, [True, False, False], [True, False, False]]
        for instance in range(NUM_ENVS):
            for t in range(3 if instance == 0 else 1):
                _, engine, reward, done, key = step(
                    instances[instance][1], instances[instance][0], int(sequence[t, instance, 0])
                )
                instances[instance] = [engine, key]
                assert_instance(out["states"][:, t], instance, engine, key)
                assert out["reward"][instance, t].item() == float(reward)
                assert out["terminated"][instance, t].item() == bool(done)
            assert_instance(out["state"], instance, *instances[instance])
        assert out["terminated"][:, 0].tolist() == [False, True, True]
        for instance in (1, 2):
            assert not out["reward"][instance, 1:].any() and not out["terminated"][instance, 1:].any()


def _compiled_steps(env) -> int:
    return sum(fn._cache_size() for (name, *_), fn in env._k._jitted.items() if name[0] in ("step", "sequence"))


def test_finished_environments_reuse_the_compiled_step():
    with se.make("CraftaxClassicSymbolic", num_envs=NUM_ENVS, device="cpu") as env:
        state = env.reset(seed=SEED)
        sequence = torch.as_tensor(actions(env, 3, SEED)).permute(1, 0, 2)
        env.step(sequence, frames=None)
        compiled = _compiled_steps(env)
        get_field(state, "engine/player_health")[0] = 0
        env.set_state(state)
        out = env.step(sequence, frames=None)
        assert not out["valid"][0, 1:].any()
        assert _compiled_steps(env) == compiled


@pytest.mark.parametrize("seed", [0, SEED, 2**31 - NUM_ENVS, 2**31 - 1, 2**31, 2**33 + 5, -2])
def test_seed_keys_equal_prng_keys(seed):
    from stable_envs.envs.craftax.kernel import Instances, Kernel

    device = jax.devices("cpu")[0]
    try:
        expected = np.stack([np.asarray(jax.random.PRNGKey(seed + i)) for i in range(NUM_ENVS)])
    except OverflowError:
        pytest.skip("JAX does not take this seed without 64-bit integers")
    np.testing.assert_array_equal(
        np.asarray(Kernel.seed_keys(seed, Instances(NUM_ENVS, NUM_ENVS, device, None))), expected
    )


def test_render_of_any_batch_equals_render_of_each_row():
    with se.make("CraftaxClassicPixels", num_envs=5, device="cpu") as env:
        env.reset(seed=SEED)
        state = env.step(actions(env, 1, SEED)[0], frames=None)["state"]
        rows = torch.cat([env.render(state[i : i + 1]) for i in range(5)])
        for n in (1, 3, 4, 5):
            assert_same_state(env.render(state[:n]), rows[:n])
        empty = env.render(state[:0])
        assert empty.shape == (0, *rows.shape[1:]) and empty.dtype == torch.uint8


def test_auto_device_takes_the_cpu_without_a_jax_gpu_backend(monkeypatch):
    from stable_envs.envs.craftax import devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(devices, "has_gpu", lambda: False)
    with se.make("CraftaxClassicSymbolic", num_envs=1, device="auto") as env:
        assert env.device == torch.device("cpu")
