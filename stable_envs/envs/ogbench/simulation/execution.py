from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_batch import OGBManipulationBatchAdapter
    from .manipulation_warp import OGBManipulationWarpAdapter
    from .maze_batch import OGBMazeBatchAdapter, OGBMazeWarpAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)

#: With ``simulation_backend="auto"``, the smallest CUDA Cube/Scene batch that steps on MuJoCo Warp; smaller CUDA
#: batches step on native ``mujoco.batch`` and return CUDA tensors (L40S, 8 cores).
MJWARP_MIN_ENVS = 256


#: Smallest CUDA maze batch that ``simulation_backend="auto"`` steps on MuJoCo Warp, per body (L40S, 8 cores).
MAZE_MJWARP_MIN_ENVS = {"point": 256, "ant": 256, "humanoid": 256}


def maze_backend(
    num_envs: int,
    device: str | torch.device,
    loco: str,
    simulation_backend: str | None = "default",
    mjwarp_min_envs: int | None = None,
) -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a maze batch; see ``resolve_backend``."""
    from ...._execution.mjbatch import resolve_backend

    threshold = MAZE_MJWARP_MIN_ENVS[loco] if mjwarp_min_envs is None else mjwarp_min_envs
    return resolve_backend(num_envs, device, simulation_backend, threshold)


def make_maze_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    loco: str = "point",
    maze: str = "large",
    simulation_backend: str | None = "default",
    mjwarp_min_envs: int | None = None,
) -> OGBMazeBatchAdapter | OGBMazeWarpAdapter:
    if maze_backend(num_envs, device, loco, simulation_backend, mjwarp_min_envs) == "mjwarp":
        from .maze_batch import OGBMazeWarpAdapter

        return OGBMazeWarpAdapter(num_envs, device, seed, loco=loco, maze=maze)
    from .maze_batch import OGBMazeBatchAdapter

    return OGBMazeBatchAdapter(num_envs, device, seed, loco=loco, maze=maze)


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
