"""OGBench process lanes match local lanes for every worker count, and a rejected State restores no lane."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter  # noqa: E402
from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter  # noqa: E402
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter  # noqa: E402
from stable_envs.envs.ogbench.simulation.maze_process import OGBMazeProcessAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402


def _inputs(batch, width, steps=8):
    rng = np.random.default_rng(5)
    actions = list(rng.uniform(-0.5, 0.5, (steps, batch, width)))
    masks = [None] + [torch.from_numpy(rng.random(batch) < 0.6) for _ in range(steps - 1)]
    return actions, masks


@pytest.mark.parametrize("body", [{}, dict(loco="ant", maze="medium")])
def test_maze_process_matches_local(body):
    other = OGBMazeAdapter(5, seed=99, **body)
    foreign = other.step(np.full((5, other.action_dim), 0.1))["state"]
    actions, masks = _inputs(5, other.action_dim)
    other.close()
    assert_process_matches_local(
        lambda: OGBMazeAdapter(5, seed=17, **body),
        lambda count: OGBMazeProcessAdapter(5, seed=17, processes=count, **body),
        actions,
        masks=masks,
        counts=(1, 2, 5),
        foreign=foreign,
    )


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_manipulation_process_matches_local(family):
    other = OGBManipulationAdapter(family, 3, seed=99)
    foreign = other.step(np.full((3, 5), 0.1))["state"]
    other.close()
    actions, masks = _inputs(3, 5, steps=6)
    assert_process_matches_local(
        lambda: OGBManipulationAdapter(family, 3, seed=17),
        lambda count: OGBManipulationProcessAdapter(family, 3, seed=17, processes=count),
        actions,
        masks=masks,
        counts=(1, 3),
        foreign=foreign,
    )


def test_maze_invalid_last_lane_restores_no_lane():
    adapter = OGBMazeProcessAdapter(5, seed=17, processes=2)
    try:

        def corrupt(state):
            state["root", "position"][0] += 0.25
            state["task", "id"][-1] = 999
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(np.zeros((5, 2))))
    finally:
        adapter.close()
