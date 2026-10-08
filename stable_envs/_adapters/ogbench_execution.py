from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from . import SimulatorAdapter
from .execution import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .ogbench_manipulation_process import OGBManipulationProcessAdapter

from .ogbench_maze_process import OGBMazeProcessAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


def make_maze_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | OGBMazeProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus)
    if count:
        return OGBMazeProcessAdapter(num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(num_envs, device, seed)


def make_manipulation_adapter(
    family: str,
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    reward_task_id: int | None,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | OGBManipulationProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus)
    if count:
        from .ogbench_manipulation_process import OGBManipulationProcessAdapter

        return OGBManipulationProcessAdapter(
            family, num_envs, device, seed, reward_task_id, processes=count, cpus=cpus
        )
    return local_factory(family, num_envs, device, seed, reward_task_id)
