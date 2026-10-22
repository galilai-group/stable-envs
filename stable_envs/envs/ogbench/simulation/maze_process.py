"""OGBench Maze lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np

from ...._execution.process import ProcessAdapter
from .maze import OGBMazeAdapter


def _lanes(loco, maze, count, seed):
    return OGBMazeAdapter(count, seed=seed, loco=loco, maze=maze)


class OGBMazeProcessAdapter(ProcessAdapter):
    """Run OGBench Maze simulators and their isolated RNGs in worker processes."""

    name = "OGBMaze"

    def __init__(self, num_envs, device="cpu", seed=0, *, processes=8, cpus=None, loco="point", maze="large"):
        super().__init__(partial(_lanes, loco, maze), num_envs, device, int(seed), processes=processes, cpus=cpus)

    def reset(self, seed=None):
        self._check_idle()
        if seed is not None:
            seed = int(seed)
            # Check the complete lane range before any worker resets.
            np.random.RandomState(seed)
            np.random.RandomState(seed + self.num_envs - 1)
        return self._reset(seed)
