"""MovingBlockWorld lanes in worker processes."""

from __future__ import annotations

from functools import partial

from ...._execution.process import ProcessAdapter
from .adapter import MovingBlockWorldAdapter


def _lanes(config, size, count, seed):
    # Lanes are seeded at reset, as in one local batch.
    return MovingBlockWorldAdapter(config, count, size)


class MiniWorldProcessAdapter(ProcessAdapter, MovingBlockWorldAdapter):
    """MovingBlockWorld source envs distributed across persistent spawned CPU processes.

    Each worker holds its lanes' GL contexts, steps them and draws their step frames into shared memory.
    State checks that need the whole batch and output-device transfer stay in the parent.
    """

    name = "MovingBlockWorld"
    validate = MovingBlockWorldAdapter.validate

    def __init__(self, config, num_envs, size, device="cpu", seed=0, *, processes=8, cpus=None):
        super().__init__(partial(_lanes, config, int(size)), num_envs, device, seed, processes=processes, cpus=cpus)
