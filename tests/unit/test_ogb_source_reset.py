"""Cube/Scene environments reset as OGBench resets, reset by reset, with the two unseeded OGBench inputs fixed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs import get_field
from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench.simulation.compatibility import ARM_START
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from tests.helpers.reference.ogbench_source_reset import differences, leaves, make
from tests.helpers.reference.semantics import assert_same_state

CLASSES = {"Cube": OGBCube, "Scene": OGBScene}
NUM_ENVS = 3


def _compare(adapter, refs):
    for instance, (env, ref) in enumerate(zip(adapter.envs, refs, strict=True)):
        diff = differences(leaves(env, env.task_rng), leaves(ref))
        assert not diff, (instance, diff)


def _reset(adapter, refs, seed):
    adapter.reset(seed)
    for instance, ref in enumerate(refs):
        ref.reset(seed=None if seed is None else seed + instance)


def _steps(adapter, refs, count, seed):
    for action in np.random.default_rng(seed).uniform(-1, 1, (count, len(refs), 5)):
        adapter.step(action)
        for instance, ref in enumerate(refs):
            ref.step(action[instance])


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_fresh_and_repeated_resets_match_ogbench(family, seed):
    adapter = OGBManipulationAdapter(family, NUM_ENVS, seed=seed)
    refs = [make(family, seed + instance) for instance in range(NUM_ENVS)]
    try:
        for ref, instance_seed in zip(refs, range(seed, seed + NUM_ENVS)):
            ref.reset(seed=instance_seed)
        _compare(adapter, refs)
        for reset_seed in (seed, None, None, seed + 5, seed):
            _reset(adapter, refs, reset_seed)
            _compare(adapter, refs)
            _steps(adapter, refs, 3, 0 if reset_seed is None else reset_seed)
            _compare(adapter, refs)
    finally:
        adapter.close()
        for ref in refs:
            ref.close()


@pytest.mark.parametrize("family", CLASSES)
def test_unseeded_resets_continue_the_task_draws(family):
    adapter = OGBManipulationAdapter(family, 1, seed=0)
    try:
        tasks = []
        for _ in range(12):
            adapter.reset()
            tasks.append(adapter.envs[0].cur_task_id)
        assert len(set(tasks)) > 1
        legacy = np.random.RandomState(0)
        assert tasks == [int(legacy.randint(1, 6)) for _ in range(13)][1:]
    finally:
        adapter.close()


@pytest.mark.parametrize("family", CLASSES)
def test_restore_reproduces_the_next_resets(family):
    a = OGBManipulationAdapter(family, NUM_ENVS, seed=5)
    b = OGBManipulationAdapter(family, NUM_ENVS, seed=99)
    refs = [make(family, 5 + instance) for instance in range(NUM_ENVS)]
    try:
        for instance, ref in enumerate(refs):
            ref.reset(seed=5 + instance)
        _reset(a, refs, None)
        _steps(a, refs, 2, 1)
        saved = a.get_state()
        b.set_state(saved)
        for seed in (None, None, 8):
            _reset(a, refs, seed)
            assert_same_state(a.get_state(), b.reset(seed))
            _compare(a, refs)
            _compare(b, refs)
            _steps(a, refs, 2, 3)
            for action in np.random.default_rng(3).uniform(-1, 1, (2, NUM_ENVS, 5)):
                b.step(action)
            assert_same_state(a.get_state(), b.get_state())
    finally:
        a.close()
        b.close()
        for ref in refs:
            ref.close()


@pytest.mark.parametrize("family", CLASSES)
def test_public_env_state_follows_ogbench(family):
    with CLASSES[family](num_envs=2, seed=3) as env:
        refs = [make(family, 3 + instance) for instance in range(2)]
        try:
            for instance, ref in enumerate(refs):
                ref.reset(seed=3 + instance)
            for seed in (None, 11):
                state = env.reset(seed)
                for instance, ref in enumerate(refs):
                    ref.reset(seed=None if seed is None else seed + instance)
                    qpos = torch.as_tensor(ref.data.qpos)
                    assert torch.equal(get_field(state, "physics/qpos")[instance], qpos)
                    assert int(get_field(state, "task/id")[instance]) == ref.cur_task_id
        finally:
            for ref in refs:
                ref.close()


@pytest.mark.parametrize("family", CLASSES)
def test_a_named_or_configured_arm_start_pins_the_arm(family):
    start = np.array([0.5, 0.05, 0.3])
    with CLASSES[family](num_envs=1, seed=4) as a, CLASSES[family](num_envs=1, seed=4) as b:
        default = a.reset(4)
        pinned = a.reset(4, {"variation_values": {ARM_START: start}})
        u = a._simulator.envs[0]
        assert np.allclose(u.data.site_xpos[u._pinch_site_id], start, atol=2e-3)
        assert not torch.equal(get_field(default, "physics/qpos"), get_field(pinned, "physics/qpos"))
        # The default reset returns to OGBench's sampled start.
        assert_same_state(default, a.reset(4))
        sampled = a.reset(4, {"variation": [ARM_START]})
        assert np.allclose(u.data.site_xpos[u._pinch_site_id], u.variation_space._walk(ARM_START).value, atol=2e-3)
        assert not torch.equal(get_field(default, "physics/qpos"), get_field(sampled, "physics/qpos"))
        # A configured anchor pins every later reset, and restore carries it.
        a.set_variation({ARM_START: start})
        b.set_state(a.get_state())
        assert_same_state(a.reset(4), b.reset(4))
        assert np.allclose(u.data.site_xpos[u._pinch_site_id], start, atol=2e-3)
        a.set_variation({ARM_START: np.mean(u._arm_sampling_bounds, axis=0)})
        assert_same_state(default, a.reset(4))


@pytest.mark.parametrize("family", CLASSES)
def test_physics_stays_bit_exact_over_long_rollouts(family):
    adapter = OGBManipulationAdapter(family, 2, seed=21)
    refs = [make(family, 21 + instance) for instance in range(2)]
    try:
        for instance, ref in enumerate(refs):
            ref.reset(seed=21 + instance)
        for chunk in range(4):
            _steps(adapter, refs, 50, chunk)
            _compare(adapter, refs)
    finally:
        adapter.close()
        for ref in refs:
            ref.close()
