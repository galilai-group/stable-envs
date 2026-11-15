from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ...._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .manipulation_batch import OGBManipulationBatchAdapter
    from .manipulation_warp import OGBManipulationWarpAdapter
    from .maze_batch import OGBMazeBatchAdapter, OGBMazeWarpAdapter

_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)

#: Shipped CUDA switch points: the smallest batch that steps on MuJoCo Warp; smaller batches step on ``mujoco.batch``.
#: The knob and the tuning cache (``stable-envs tune``) take precedence.
#: Measured 2026-11-14 on Oscar RTX 3090 (EPYC 7532) and RTX A5000 (EPYC 7532 and 7662) nodes, 8 cores each: the
#: per-call step time of both backends, timed as ``stable-envs tune`` times them, at batches 1, 2, 4, ...
#: (``python -m benchmarks.backend_switch <Env>``). Where the two GPU types disagree, the larger value is kept, so
#: both hosts stay on ``mujoco.batch`` where it was faster on either one.
MAZE_CUDA_MIN_ENVS: dict[str, int | None] = {
    "point": 16,  # RTX 3090 8 (Medium), 2 (Giant); RTX A5000 16, 16
    "ant": 128,  # 64, 64; 128, 128
    "humanoid": 128,  # 64, 64; 128, 64
}


def maze_backend(
    num_envs: int,
    device: str | torch.device,
    loco: str,
    simulation_backend: str | None = "auto",
    maze: str = "large",
) -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a maze batch; see ``mujoco_backend``."""
    import torch

    from ...._execution.backends import measure_adapter
    from ...._execution.mjbatch import mujoco_backend
    from .maze import BODIES

    def build(backend: str, n: int):
        return make_maze_adapter(n, device, 0, loco=loco, maze=maze, simulation_backend=backend)

    dim = BODIES[loco][2]
    measure = measure_adapter(build, lambda n: torch.rand(n, dim, device=device) * 2 - 1)
    path = f"ogb-sim/{loco}maze-{maze}"
    return mujoco_backend(
        path, num_envs, device, simulation_backend, cuda_min_envs=MAZE_CUDA_MIN_ENVS[loco], measure=measure
    )


def make_maze_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    loco: str = "point",
    maze: str = "large",
    simulation_backend: str | None = "auto",
) -> OGBMazeBatchAdapter | OGBMazeWarpAdapter:
    if maze_backend(num_envs, device, loco, simulation_backend, maze) == "mjwarp":
        from .maze_batch import OGBMazeWarpAdapter

        return OGBMazeWarpAdapter(num_envs, device, seed, loco=loco, maze=maze)
    from .maze_batch import OGBMazeBatchAdapter

    return OGBMazeBatchAdapter(num_envs, device, seed, loco=loco, maze=maze)


#: Shipped CUDA switch points of Cube and Scene: the smallest batch that steps on MuJoCo Warp; smaller batches step
#: on native ``mujoco.batch``. The knob and the tuning cache (``stable-envs tune``) take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
MANIPULATION_CUDA_MIN_ENVS: dict[str, int] = {"Cube": 64, "Scene": 128}


def _measure_manipulation(env_type, device):
    """Build and step ``env_type`` on one backend, for ``stable-envs tune``; None when the caller has no class."""
    from ...._execution.backends import measure_env

    if env_type is None:
        return None
    return measure_env(lambda backend, n: env_type(num_envs=n, device=device, simulation_backend=backend))


def manipulation_backend(
    family: str,
    num_envs: int,
    device: str | torch.device,
    simulation_backend: str | None = "auto",
    env_type: type | None = None,
) -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` for a Cube/Scene batch; see ``select_backend``."""
    import torch

    from ...._execution.backends import Rungs, select_backend

    if simulation_backend == "mjwarp" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='mjwarp' requires device='cuda'")
    return select_backend(
        manipulation_path(family),
        num_envs,
        device,
        simulation_backend,
        {"cpu": Rungs("native", "native"), "cuda": Rungs("native", "mjwarp", MANIPULATION_CUDA_MIN_ENVS.get(family))},
        choices=("native", "mjwarp"),
        measure=_measure_manipulation(env_type, device),
    )


def manipulation_path(family: str) -> str:
    """The ``select_backend`` path of a Cube or Scene batch."""
    return f"ogb-sim/{family.lower()}"


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
