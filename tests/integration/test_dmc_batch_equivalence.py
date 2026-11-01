"""DMC batch lanes match local lanes at every thread count, through implicit resets, restores and physics Variation."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")
pytest.importorskip("mujoco.batch")

from stable_envs.envs.dmcontrol.simulation import first_family  # noqa: E402
from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_batch, make_local  # noqa: E402
from stable_envs.envs.dmcontrol.simulation.physics import LEAVES  # noqa: E402
from tests.helpers.process_equivalence import assert_process_matches_local  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402


def test_local_lanes_compile_once_and_own_model_and_data(monkeypatch):
    builder = Mock(wraps=first_family._make_first_family)
    monkeypatch.setattr(first_family, "_make_first_family", builder)
    adapter = make_local("cartpole", 3, seed=17)
    try:
        lanes = adapter.envs
        assert builder.call_count == 1
        assert len({id(lane.task) for lane in lanes}) == 3
        spec, clone_spec = lanes[0].action_spec(), lanes[1].action_spec()
        assert (clone_spec.shape, clone_spec.dtype) == (spec.shape, spec.dtype)
        np.testing.assert_array_equal(clone_spec.minimum, spec.minimum)
        np.testing.assert_array_equal(clone_spec.maximum, spec.maximum)
        before = [(lane.physics.model.body_mass.copy(), lane.physics.data.qpos.copy()) for lane in lanes]
        lanes[1].physics.model.body_mass[0] += 1
        lanes[1].physics.data.qpos[0] += 1
        for lane, (body_mass, qpos) in zip(lanes[::2], before[::2]):
            np.testing.assert_array_equal(lane.physics.model.body_mass, body_mass)
            np.testing.assert_array_equal(lane.physics.data.qpos, qpos)
    finally:
        adapter.close()


@pytest.mark.parametrize("num_envs", [0, -1])
def test_local_lanes_refuse_an_empty_batch(num_envs):
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        make_local("cartpole", num_envs)


def _foreign_and_actions(domain):
    """A State whose lanes reach the time limit in the first or the second control step of the next actions."""
    other = make_local(domain, 5, seed=99)
    try:
        other.step(np.full((5, other.action_dim), 0.2))
        foreign = other.get_state()
        limit = math.ceil(other.envs[0]._env._step_limit)
        actions = list(np.random.default_rng(5).uniform(-1, 1, (8, 5, other.action_dim)))
    finally:
        other.close()
    foreign["lifecycle", "step_count"] = limit - torch.tensor([[2], [3], [2], [3], [2]])
    return foreign, actions


@pytest.mark.parametrize("domain", sorted(FACTORIES))
def test_batch_matches_local(domain):
    foreign, actions = _foreign_and_actions(domain)
    assert_process_matches_local(
        lambda: make_local(domain, 5, seed=17),
        lambda threads: make_batch(domain, 5, seed=17, threads=threads),
        actions,
        counts=(1, 2, 5),
        foreign=foreign,
    )


@pytest.mark.parametrize("domain", ["walker", "ball_in_cup"])
def test_batch_matches_local_under_physics_variation(domain):
    rng = np.random.default_rng(3)
    values = {leaf.path: rng.uniform(leaf.low, leaf.high, 5) for leaf in LEAVES[domain]}

    def varied(adapter):
        adapter.configure_physics(values)
        adapter.reset(seed=17)
        return adapter

    foreign, actions = _foreign_and_actions(domain)
    assert_process_matches_local(
        lambda: varied(make_local(domain, 5, seed=17)),
        lambda threads: varied(make_batch(domain, 5, seed=17, threads=threads)),
        actions,
        counts=(1, 5),
        foreign=foreign,
    )


def test_batch_matches_local_when_physics_changes_mid_episode():
    rng = np.random.default_rng(3)
    values = {leaf.path: rng.uniform(leaf.low, leaf.high, 5) for leaf in LEAVES["walker"]}
    # After 15 zero actions every lane touches the floor, so the change lands during contact.
    actions = [np.zeros((5, 6))] * 15 + list(rng.uniform(-1, 1, (4, 5, 6)))
    for threads in (1, 5):
        local, batch = make_local("walker", 5, seed=17), make_batch("walker", 5, seed=17, threads=threads)
        try:
            for t, action in enumerate(actions):
                if t == 15:
                    local.configure_physics(values)
                    batch.configure_physics(values)
                try:
                    assert_same_state(batch.step(action), local.step(action))
                except AssertionError as error:
                    raise AssertionError(f"threads={threads}, step {t}: {error}") from error
        finally:
            local.close()
            batch.close()


def test_close_is_idempotent_and_releases_the_batch():
    adapter = make_batch("cartpole", 2, threads=1)
    adapter.close()
    adapter.close()
    assert adapter.b is None and adapter.task is None
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()
