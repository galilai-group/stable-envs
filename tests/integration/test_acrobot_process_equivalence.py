"""Acrobot process lanes match local lanes for every worker count, and a rejected State restores no lane."""

import numpy as np
import torch

from stable_envs.envs.classic_control.simulation.local import ClassicControlAdapter
from stable_envs.envs.classic_control.simulation.process import AcrobotProcessAdapter
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local


def _local(seed):
    adapter = ClassicControlAdapter("acrobot", 5)
    adapter.reset(seed)
    return adapter


def test_process_matches_local():
    rng = np.random.default_rng(5)
    actions = list(rng.integers(0, 3, (10, 5, 1)))
    masks = [None] + [torch.from_numpy(rng.random(5) < 0.6) for _ in range(9)]
    other = _local(99)
    other.configure({"link_mass_1": np.linspace(0.6, 1.4, 5)})
    foreign = other.step(np.ones((5, 1), np.int64))["state"]
    other.close()
    assert_process_matches_local(
        lambda: _local(17),
        lambda count: AcrobotProcessAdapter("acrobot", 5, seed=17, processes=count),
        actions,
        masks=masks,
        counts=(1, 2, 5),
        foreign=foreign,
    )


def test_invalid_last_lane_rng_restores_no_lane():
    adapter = AcrobotProcessAdapter("acrobot", 5, seed=17, processes=2)
    try:

        def corrupt(state):
            state["theta1"][0] += 0.25
            state["rng", "state"][-1].fill_(-1)
            return state

        assert_atomic_restore(
            adapter, corrupt, (ValueError, OverflowError), lambda a: a.step(np.ones((5, 1), np.int64))
        )
    finally:
        adapter.close()
