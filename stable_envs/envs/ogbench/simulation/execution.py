from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ...._execution.policy import cpu_affinity, process_count
from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_process import OGBManipulationProcessAdapter

from .maze_process import OGBMazeProcessAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)

#: Fewest lanes that use workers when the ``worker_min_lanes`` knob is not set; measured on 8 cores.
MAZE_MIN_LANES = 2
#: Cube and Scene workers with one lane each step slower than workers with two.
MANIPULATION_MIN_LANES, MANIPULATION_LANES_PER_WORKER = 4, 2


def make_maze_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
    *,
    loco: str = "point",
    maze: str = "large",
) -> _AdapterT | OGBMazeProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="ogb-sim", min_lanes=MAZE_MIN_LANES)
    if count:
        return OGBMazeProcessAdapter(num_envs, device, seed, processes=count, cpus=cpus, loco=loco, maze=maze)
    return local_factory(num_envs, device, seed, loco=loco, maze=maze)


def make_manipulation_adapter(
    family: str,
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    reward_task_id: int | None,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | OGBManipulationProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(
        num_envs,
        cpus,
        path="ogb-sim",
        min_lanes=MANIPULATION_MIN_LANES,
        lanes_per_worker=MANIPULATION_LANES_PER_WORKER,
    )
    if count:
        from .manipulation_process import OGBManipulationProcessAdapter

        return OGBManipulationProcessAdapter(
            family, num_envs, device, seed, reward_task_id, processes=count, cpus=cpus
        )
    return local_factory(family, num_envs, device, seed, reward_task_id)
