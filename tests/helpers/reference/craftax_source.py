"""Pinned SWM oracle and recorded research inputs, exclusively for Craftax tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "tests/fixtures/craftax_source"
NAMES = ("CraftaxClassicPixels", "CraftaxClassicSymbolic", "CraftaxPixels", "CraftaxSymbolic")
ACTIONS = {
    "mine_tree": 5,
    "craft": 11,
    "combat": 5,
    "damage": 0,
    "night": 0,
    "terminal": 0,
    "near_limit": 0,
    "stochastic_grass": 5,
}


def oracle(name):
    root = Path(os.environ.get("SWM_ROOT") or Path(__file__).resolve().parents[3].parent / "stable-worldmodel")
    source = root / "stable_worldmodel/envs/craftax/craftax.py"
    if not source.exists():
        import pytest

        pytest.skip(f"no stable-worldmodel checkout at {root}; set SWM_ROOT to compare against the Craftax source")
    spec = importlib.util.spec_from_file_location("craftax_test_oracle", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name + "Wrapper")(
        backend=os.environ.get("CRAFTAX_JAX_BACKEND", "cpu"), render_mode="rgb_array"
    )


def fixture(env, name, case):
    from stable_envs.envs.craftax.simulation import decode

    state = env.get_state()
    with np.load(EVIDENCE / (name + "Wrapper-" + case + ".npz"), allow_pickle=False) as arrays:
        for key in state.keys(True, True):
            flat = "/".join(key)
            if flat in arrays:
                state[key] = torch.from_numpy(arrays[flat].copy()).to(device=env.device, dtype=state[key].dtype)
    a = env._simulator
    row = {k: state[("engine", *k)][0].cpu().numpy() for k in a.engine}
    decoded = decode(a.template, row)
    done = bool(np.asarray(a.terminal_fn(decoded)))
    if a.retains_symbolic_output:
        state["source_cache", "last_symbolic"] = torch.from_numpy(np.asarray(a.output_fn(decoded)).copy())[None].to(
            env.device
        )
    state["lifecycle", "terminated"].fill_(done)
    return state


def restore_oracle(source, env, state):
    import jax.numpy as jnp

    from stable_envs.envs.craftax.simulation import decode

    source._state = decode(source._state, {k: state[("engine", *k)][0].cpu().numpy() for k in env._simulator.engine})
    source._key = jnp.asarray(state["rng", "wrapper_key"][0].cpu().numpy().astype(np.uint32))
    if env._simulator.retains_symbolic_output:
        source._last_obs = state["source_cache", "last_symbolic"][0].cpu().numpy().copy()


def assert_engine(source, env, state):
    from stable_envs.envs.craftax.simulation import numeric_leaves

    for key, value in numeric_leaves(source._state):
        np.testing.assert_array_equal(state[("engine", *key)][0].cpu().numpy(), value, err_msg="/".join(key))
    np.testing.assert_array_equal(state["rng", "wrapper_key"][0].cpu().numpy(), np.asarray(source._key))


def assert_state_equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert torch.equal(a[key], b[key]), key


def symbolic_output(env, lane=0, state=None):
    """Test-side learner output selection; full symbolic retains the source cache."""
    from stable_envs.envs.craftax.simulation import decode

    state = env.get_state() if state is None else state
    if env._simulator.retains_symbolic_output:
        return state["source_cache", "last_symbolic"][lane].cpu().numpy().copy()
    row = {k: state[("engine", *k)][lane].cpu().numpy() for k in env._simulator.engine}
    return np.asarray(env._simulator.output_fn(decode(env._simulator.template, row)))
