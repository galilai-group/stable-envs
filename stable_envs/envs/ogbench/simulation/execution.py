from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_batch import OGBManipulationBatchAdapter
    from .manipulation_warp import OGBManipulationWarpAdapter
    from .maze_batch import OGBMazeBatchAdapter, OGBMazeWarpAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


def maze_backend(device: str | torch.device, simulation_backend: str | None = "auto") -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a maze batch; see ``mujoco_backend``."""
    from ...._execution.mjbatch import mujoco_backend

    return mujoco_backend(device, simulation_backend)


def make_maze_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    loco: str = "point",
    maze: str = "large",
    simulation_backend: str | None = "auto",
) -> OGBMazeBatchAdapter | OGBMazeWarpAdapter:
    if maze_backend(device, simulation_backend) == "mjwarp":
        from .maze_batch import OGBMazeWarpAdapter

        return OGBMazeWarpAdapter(num_envs, device, seed, loco=loco, maze=maze)
    from .maze_batch import OGBMazeBatchAdapter

    return OGBMazeBatchAdapter(num_envs, device, seed, loco=loco, maze=maze)


def manipulation_backend(device: str | torch.device, simulation_backend: str | None = "auto") -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a Cube/Scene batch; see ``mujoco_backend``."""
    from ...._execution.mjbatch import mujoco_backend

    return mujoco_backend(device, simulation_backend)


def make_manipulation_adapter(
    family: str,
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    reward_task_id: int | None,
    simulation_backend: str | None = "auto",
) -> OGBManipulationBatchAdapter | OGBManipulationWarpAdapter:
    """Cube/Scene environments on the backend that ``manipulation_backend`` selects: MuJoCo Warp or ``mujoco.batch``."""
    backend = manipulation_backend(device, simulation_backend)
    if backend == "mjwarp":
        from .manipulation_warp import OGBManipulationWarpAdapter

        return OGBManipulationWarpAdapter(family, num_envs, device, seed, reward_task_id)
    from .manipulation_batch import OGBManipulationBatchAdapter

    return OGBManipulationBatchAdapter(family, num_envs, device, seed, reward_task_id)
