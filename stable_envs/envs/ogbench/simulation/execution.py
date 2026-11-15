from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_batch import OGBManipulationBatchAdapter
    from .manipulation_warp import OGBManipulationWarpAdapter
    from .maze_batch import OGBMazeBatchAdapter, OGBMazeWarpAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)

#: Smallest CUDA maze batch that ``simulation_backend="auto"`` steps on MuJoCo Warp, per body: the smallest measured
#: batch (of 1, 8, 64, 256, 1024, 4096; L40S, 8 cores) at which MuJoCo Warp steps faster than mujoco.batch.
MAZE_MJWARP_MIN_ENVS = {"point": 4096, "ant": 256, "humanoid": 256}


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


#: Shipped CUDA switch points of Cube and Scene: the smallest batch that steps on MuJoCo Warp; smaller batches step
#: on native ``mujoco.batch``. The knob and the tuning cache (``stable-envs tune``) take precedence.
#: Measured 2026-11-14 on Oscar RTX 3090 (EPYC 7532) and RTX A5000 (EPYC 7532 and 7662) nodes, 8 cores each: the
#: per-call step time of both backends, timed as ``stable-envs tune`` times them, at batches 1, 2, 4, ..., 4096
#: (``python -m benchmarks.backend_switch <Env>``). Where the two GPU types disagree, the larger value is kept, so
#: both hosts stay on the CPU backend where it was faster on either one.
MANIPULATION_CUDA_MIN_ENVS: dict[str, int] = {"Cube": 64, "Scene": 128}


def _measure_manipulation(family: str, device: str | torch.device):
    """Build and step Cube/Scene lanes on one backend, for ``stable-envs tune``."""
    import torch

    from ...._execution.backends import measure_adapter

    return measure_adapter(
        lambda backend, n: make_manipulation_adapter(family, n, device, 0, None, backend),
        lambda n: torch.rand(n, 5, device=device) * 2 - 1,
    )


def manipulation_backend(
    family: str, num_envs: int, device: str | torch.device, simulation_backend: str | None = "auto"
) -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a Cube/Scene batch; see ``select_backend``."""
    import torch

    from ...._execution.backends import Rungs, select_backend

    if simulation_backend == "mjwarp" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='mjwarp' requires device='cuda'")
    return select_backend(
        f"ogb-sim/{family.lower()}",
        num_envs,
        device,
        simulation_backend,
        {"cpu": Rungs("native", "native"), "cuda": Rungs("native", "mjwarp", MANIPULATION_CUDA_MIN_ENVS.get(family))},
        choices=("native", "mjwarp"),
        measure=_measure_manipulation(family, device),
    )


def make_manipulation_adapter(
    family: str,
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    reward_task_id: int | None,
    simulation_backend: str | None = "auto",
) -> OGBManipulationBatchAdapter | OGBManipulationWarpAdapter:
    """Cube/Scene lanes on the backend that ``manipulation_backend`` selects: MuJoCo Warp or ``mujoco.batch``."""
    backend = manipulation_backend(family, num_envs, device, simulation_backend)
    if backend == "mjwarp":
        from .manipulation_warp import OGBManipulationWarpAdapter

        return OGBManipulationWarpAdapter(family, num_envs, device, seed, reward_task_id)
    from .manipulation_batch import OGBManipulationBatchAdapter

    return OGBManipulationBatchAdapter(family, num_envs, device, seed, reward_task_id)
