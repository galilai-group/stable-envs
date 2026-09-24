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


def test_full_symbolic_preserves_source_values_outside_declared_box():
    # SWM declares [0,1], but upstream includes books/2 and enchantment enums.
    # This is an output-contract finding, not permission to clamp observations.
    source = oracle("CraftaxSymbolic")
    with se.make("CraftaxSymbolic", device=DEVICE) as env:
        source.reset(seed=17)
        state = env.reset(seed=17)
        state["engine", "inventory", "books"].fill_(4)
        state["engine", "sword_enchantment"].fill_(2)
        env.set_state(state)
        restore_oracle(source, env, state)
        expected, _, _, _, _ = source.step(0)
        env.step([0])
        actual = np.asarray(env._simulator.output_fn(env._simulator.states[0]))
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (8268,) and actual.dtype == np.float32
        assert actual.max() >= 2


@pytest.mark.parametrize("name", NAMES)
def test_batch_matches_three_independent_source_instances(name):
    sources = [oracle(name) for _ in range(3)]
    with se.make(name, num_envs=3, device=DEVICE) as env:
        state = env.reset(seed=17)
        for lane, source in enumerate(sources):
            source.reset(seed=17 + lane)
            assert_engine(source, env, state[lane : lane + 1])
        for actions in [[0, 5, 1], [2, 0, 3], [5, 4, 0]]:
            expected = [source.step(action) for source, action in zip(sources, actions, strict=True)]
            result = env.step(actions)
            for lane, source in enumerate(sources):
                assert_engine(source, env, result["state"][lane : lane + 1])
                assert float(result["reward"][lane]) == expected[lane][1]
                if "Symbolic" in name:
                    np.testing.assert_array_equal(
                        np.asarray(env._simulator.output_fn(env._simulator.states[lane])), expected[lane][0]
                    )


@pytest.mark.parametrize("name", NAMES)
def test_invalid_source_invariants_leave_all_lanes_unchanged(name):
    with se.make(name, num_envs=3, device=DEVICE) as env:
        good = env.reset(seed=17)
        for key, value in [
            (("rng", "wrapper_key"), 2**32),
            (("engine", "state_rng"), -1),
            (("engine", "player_position"), 999),
            (("engine", "player_direction"), 0),
            (("engine", "inventory", "wood"), 100),
            (("engine", "growing_plants_age"), -1),
        ]:
            bad = good.clone()
            bad[key][-1].fill_(value)
            with pytest.raises(ValueError):
                env.set_state(bad)
            assert_state_equal(good, env.get_state())
        bad = good.clone()
        bad["engine", "mob_map"][-1].fill_(True)
        with pytest.raises(ValueError):
            env.set_state(bad)
        assert_state_equal(good, env.get_state())
        if not env.CLASSIC:
            bad = good.clone()
            bad["engine", "potion_mapping"][-1].fill_(0)
            with pytest.raises(ValueError):
                env.set_state(bad)
            assert_state_equal(good, env.get_state())


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxSymbolic"])
def test_symbolic_order_chunk_nonmutation_and_exhaustive_restore(name, tmp_path):
    import jax

    from stable_envs._adapters.craftax import decode

    with se.make(name, num_envs=3, device=DEVICE) as env:
        state = env.reset(seed=17)
        snapshots = state.clone()

        def output(batch, chunk):
            _, decoded = env._simulator.validate(batch)
            values = []
            for start in range(0, len(decoded), chunk):
                values.extend(np.asarray(env._simulator.output_fn(s)) for s in decoded[start : start + chunk])
            return np.stack(values)

        expected = output(state, 3)
        np.testing.assert_array_equal(expected, output(state, 1))
        np.testing.assert_array_equal(expected, output(state, 2))
        np.testing.assert_array_equal(expected[[2, 0, 1]], output(state[[2, 0, 1]], 2))
        assert expected.dtype == np.float32
        assert expected.shape == (3, 1345 if env.CLASSIC else 8268)
        assert_state_equal(state, snapshots)
        assert_state_equal(state, env.get_state())
        with pytest.raises(TypeError):
            env.set_state(torch.from_numpy(expected))
        assert_state_equal(state, env.get_state())
        # Terminal symbolic output remains exact and does not consume either key.
        source = oracle(name)
        source.reset(seed=999)
        with se.make(name, device=DEVICE) as single:
            single.reset(seed=999)
            terminal = fixture(single, name, "terminal")
        restore_oracle(source, env, terminal)
        eager_terminal = np.asarray(source._env.get_obs(source._state))
        expected_terminal = np.asarray(
            jax.jit(source._env.get_obs, backend=os.environ.get("CRAFTAX_JAX_BACKEND", "cpu"))(source._state)
        )
        row = {k: terminal[("engine", *k)][0].cpu().numpy() for k in env._simulator.engine}
        actual_terminal = np.asarray(env._simulator.output_fn(decode(env._simulator.template, row)))
        np.savez_compressed(
            tmp_path / (name + "-terminal-paths.npz"),
            eager=eager_terminal,
            source_jit=expected_terminal,
            candidate_jit=actual_terminal,
            **{"state/" + "/".join(k): v.cpu().numpy() for k, v in terminal.items(True, True)},
        )
        np.testing.assert_array_equal(expected_terminal, actual_terminal)
        assert_state_equal(state, env.get_state())
    assert env._simulator.states is None and env._simulator.keys == []
