"""Fetch process lanes match local lanes for every worker count, and a rejected State restores no lane."""

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

from stable_envs.envs.fetch.simulation.adapter import FetchAdapter  # noqa: E402
from stable_envs.envs.fetch.simulation.process import FetchProcessAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402


@pytest.mark.parametrize("task", ["Reach", "PickAndPlace"])
def test_process_matches_local(task):
    rng = np.random.default_rng(5)
    actions = list(rng.uniform(-1, 1, (10, 5, 4)))
    masks = [None] + [torch.from_numpy(rng.random(5) < 0.6) for _ in range(9)]
    other = FetchAdapter(5, seed=99, task=task)
    foreign = other.step(np.full((5, 4), 0.3))["state"]
    other.close()
    assert_process_matches_local(
        lambda: FetchAdapter(5, seed=17, task=task),
        lambda count: FetchProcessAdapter(5, seed=17, task=task, processes=count),
        actions,
        masks=masks,
        counts=(1, 2, 5),
        foreign=foreign,
    )


def test_invalid_last_lane_restores_no_lane():
    adapter = FetchProcessAdapter(5, task="Push", processes=2)
    try:

        def corrupt(state):
            state["qpos"][0, 0] += 0.01
            state["rng", "has_uint32"][-1] = 2
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(np.zeros((5, 4))))
    finally:
        adapter.close()
