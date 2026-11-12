from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ...._execution.policy import cpu_affinity, process_count
from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_batch import OGBManipulationBatchAdapter
    from .manipulation_warp import OGBManipulationWarpAdapter

from .maze_process import OGBMazeProcessAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)

#: With ``simulation_backend="auto"``, the smallest CUDA Cube/Scene batch that steps on MuJoCo Warp; smaller CUDA
#: batches step on native ``mujoco.batch`` and return CUDA tensors (L40S, 8 cores).
MJWARP_MIN_ENVS = 256


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
    simulation_backend: str | None = "default",
    mjwarp_min_envs: int = MJWARP_MIN_ENVS,
) -> OGBManipulationBatchAdapter | OGBManipulationWarpAdapter:
    """Cube/Scene lanes on the backend that ``resolve_backend`` selects: MuJoCo Warp or native ``mujoco.batch``."""
    from ...._execution.mjbatch import resolve_backend

    backend = resolve_backend(num_envs, device, simulation_backend, mjwarp_min_envs)
    if backend == "mjwarp":
        from .manipulation_warp import OGBManipulationWarpAdapter

        return OGBManipulationWarpAdapter(family, num_envs, device, seed, reward_task_id)
    from .manipulation_batch import OGBManipulationBatchAdapter

    return OGBManipulationBatchAdapter(family, num_envs, device, seed, reward_task_id)
