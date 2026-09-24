"""Independent source parity and atomic exhaustive-State tests for all four registrations."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytest.importorskip("craftax")
pytest.importorskip("jax")

import stable_envs as se
from tests.helpers.reference.craftax_source import (
    ACTIONS,
    NAMES,
    assert_engine,
    assert_state_equal,
    fixture,
    oracle,
    restore_oracle,
)

DEVICE = os.environ.get("CRAFTAX_DEVICE", "cpu")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_source_reset_all_actions_and_reset_replay(name, seed):
    source = oracle(name)
    with se.make(name, device=DEVICE) as env:
        source.reset(seed=seed)
        state = env.reset(seed=seed)
        assert_engine(source, env, state)
        for action in range(source.action_space.n):
            output, reward, done, truncated, _ = source.step(action)
            actual = env.step(torch.tensor([action], dtype=torch.int32))
            assert_engine(source, env, actual["state"])
            assert float(actual["reward"][0]) == reward
            assert bool(actual["terminated"][0]) == done
            assert bool(actual["truncated"][0]) == truncated
            # Learner output stays upstream; prove a symbolic projection of complete State is exact.
            if "Symbolic" in name:
                _, restored = env._simulator.validate(actual["state"])
                np.testing.assert_array_equal(np.asarray(env._simulator.output_fn(restored[0])), output)
            if done:
                break
        snap = env.get_state()
        assert_state_equal(snap, env.get_state())
        source.reset()
        actual = env.reset()
        assert_engine(source, env, actual)
        env.set_state(snap)
        assert_state_equal(actual, env.reset())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("case", list(ACTIONS))
def test_recorded_gameplay_different_history_continuation(name, case):
    source = oracle(name)
    with se.make(name, device=DEVICE) as env:
        source.reset(seed=999)
        env.reset(seed=999)
        for action in [1, 3, 5]:
            env.step([action])
            source.step(action)
        state = fixture(env, name, case)
        restore_oracle(source, env, state)
        env.set_state(state)
        assert_state_equal(state, env.get_state())
        if bool(state["lifecycle", "terminated"][0]):
            with pytest.raises(RuntimeError, match="terminal"):
                env.step([0])
            # Explicit terminal source State remains readable and restorable.
            np.testing.assert_array_equal(
                np.asarray(source._env.get_obs(source._state)),
                np.asarray(env._simulator.upstream.get_obs(env._simulator.states[0])),
            )
            return
        output, reward, done, truncated, _ = source.step(ACTIONS[case])
        result = env.step([ACTIONS[case]])
        assert_engine(source, env, result["state"])
        assert float(result["reward"][0]) == reward
        assert bool(result["terminated"][0]) == done
        assert bool(result["truncated"][0]) == truncated
        if "Symbolic" in name:
            np.testing.assert_array_equal(np.asarray(env._simulator.output_fn(env._simulator.states[0])), output)
        frozen = result["state"].clone()
        env.reset(seed=42)
        env.set_state(frozen)
        assert_state_equal(frozen, env.get_state())


@pytest.mark.parametrize("name", NAMES)
def test_batched_restore_is_atomic_and_owning(name):
    with se.make(name, num_envs=3, device=DEVICE) as env:
        original = env.reset(seed=17)
        bad_states = []
        for key in [("meta", "signature"), ("engine", "map"), ("static", "map_size", "0")]:
            bad = original.clone()
            if key == ("engine", "map"):
                bad[key][-1].fill_(999)
            else:
                bad[key][-1].add_(1)
            bad_states.append(bad)
        bad = original.clone()
        bad["rng", "wrapper_key"] = bad["rng", "wrapper_key"].to(torch.float64)
        bad_states.append(bad)
        bad = original.clone()
        bad["lifecycle", "terminated"][-1] = True
        bad_states.append(bad)
        bad = original.clone()
        bad["engine", "light_level"][-1] = float("nan")
        bad_states.append(bad)
        bad = original.clone()
        del bad["engine", "state_rng"]
        bad_states.append(bad)
        bad = original.clone()
        bad["unexpected"] = torch.zeros(3, device=DEVICE)
        bad_states.append(bad)
        for bad in bad_states:
            with pytest.raises((ValueError, TypeError)):
                env.set_state(bad)
            assert_state_equal(original, env.get_state())
        with pytest.raises(ValueError):
            env.set_state(original[:2])
        for action in [[-1, 0, 0], [0.5, 0, 0], [0, 0, 999], [True, False, True], [0, 0]]:
            with pytest.raises(ValueError):
                env.step(action)
            assert_state_equal(original, env.get_state())
        result = env.step([1, 2, 3])["state"]
        env.set_state(original)
        assert_state_equal(result, env.transition(original, torch.tensor([1, 2, 3])))
        assert_state_equal(original, env.get_state())
        # Input owns its storage, and restoration does not retain it.
        restored = original.clone()
        env.set_state(restored)
        restored["engine", "map"].fill_(0)
        assert_state_equal(original, env.get_state())
        assert not torch.equal(restored["engine", "map"], env.get_state()["engine", "map"])


@pytest.mark.parametrize("name", NAMES)
def test_independent_instances_and_step_never_calls_set_state(name, monkeypatch):
    with se.make(name, device=DEVICE) as a, se.make(name, device=DEVICE) as b:
        a.reset(seed=17)
        b.reset(seed=42)
        snapshot = a.get_state()
        b.set_state(snapshot)

        def forbidden(*args):
            raise AssertionError("ordinary step must not restore")

        monkeypatch.setattr(a._simulator, "set_state", forbidden)
        for action in [5, 0, 1, 2]:
            assert_state_equal(a.step([action])["state"], b.step([action])["state"])
        a.reset(seed=2)
        before = b.get_state()
        a.step([0])
        assert_state_equal(before, b.get_state())
        with pytest.raises((TypeError, ValueError)):
            b.set_state(torch.zeros((1, 1345)))
        with pytest.raises(ValueError):
            b.set_variation(day_length=10)
