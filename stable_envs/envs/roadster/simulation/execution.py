"""Choose the physics backend of a Roadster batch: native ``mujoco.batch`` on CPU or MuJoCo Warp on CUDA."""

from __future__ import annotations

import torch

from ...._execution.backends import measure_env
from ...._execution.mjbatch import mujoco_backend

#: Shipped CUDA switch points: the smallest batch that steps on MuJoCo Warp; smaller batches step on ``mujoco.batch``.
#: The knob and the tuning cache (``stable-envs tune``) take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
CUDA_MIN_ENVS: dict[str, int | None] = {
    "loop": 32,  # RTX 3090 16, RTX A5000 32
    "mission": 16,
    "traffic": 32,  # 16, 32
    "rooftop": 64,  # 32, 64
    "carhop": 32,  # 16, 32
}


def roadster_path(task: str) -> str:
    """The simulation path of a Roadster task, which keys its switch point in the tuning cache."""
    return f"roadster-sim/{task}"


def roadster_backend(
    num_envs: int,
    device: str | torch.device,
    simulation_backend: str | None = "auto",
    task: str = "traffic",
    env_type: type | None = None,
) -> str:
    """Return ``"native"`` or ``"mjwarp"`` for a batch.

    ``"auto"`` steps on the native batch on CPU. On CUDA it steps small batches on the native batch and large ones
    on MuJoCo Warp; see ``mujoco_backend``.
    """
    measure = (
        None
        if env_type is None
        else measure_env(lambda backend, n: env_type(num_envs=n, device=device, simulation_backend=backend))
    )
    path = roadster_path(task)
    min_envs = CUDA_MIN_ENVS[task]
    return mujoco_backend(path, num_envs, device, simulation_backend, cuda_min_envs=min_envs, measure=measure)


def make_roadster_adapter(task: str, num_envs: int, device="cpu", backend: str = "native"):
    """The adapter of ``backend`` for ``num_envs`` environments of ``task``, with State on ``device``."""
    if backend == "mjwarp":
        from .warp import RoadsterWarpAdapter

        return RoadsterWarpAdapter(task, num_envs, device)
    from .batch import RoadsterNativeAdapter

    return RoadsterNativeAdapter(task, num_envs, device)
