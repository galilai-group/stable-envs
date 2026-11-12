from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ...._execution.policy import cpu_affinity, process_count
from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .cube_mjx import CubeMJXAdapter
    from .manipulation_process import OGBManipulationProcessAdapter

from .maze_process import OGBMazeProcessAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


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
    count = process_count(num_envs, cpus, path="ogb-sim")
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
) -> _AdapterT | CubeMJXAdapter | OGBManipulationProcessAdapter:
    import torch

    if family == "Cube" and torch.device(device).type == "cuda":
        from .cube_mjx import CubeMJXAdapter

        return CubeMJXAdapter(family, num_envs, device, seed, reward_task_id)
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="ogb-sim")
    if count:
        from .manipulation_process import OGBManipulationProcessAdapter

        return OGBManipulationProcessAdapter(
            family, num_envs, device, seed, reward_task_id, processes=count, cpus=cpus
        )
    return local_factory(family, num_envs, device, seed, reward_task_id)
