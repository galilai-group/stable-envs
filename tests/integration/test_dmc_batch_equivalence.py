"""DMC batch lanes match local lanes at every thread count, through implicit resets, restores and physics Variation."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")
pytest.importorskip("mujoco.batch")

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_batch, make_local  # noqa: E402
from stable_envs.envs.dmcontrol.simulation.physics import LEAVES  # noqa: E402
from tests.helpers.process_equivalence import assert_process_matches_local  # noqa: E402


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
