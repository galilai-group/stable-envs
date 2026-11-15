"""Choose the physics backend of a Roadster batch: native ``mujoco.batch`` on CPU or MuJoCo Warp on CUDA."""

from __future__ import annotations

import torch

from ...._execution.backends import measure_adapter
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


def roadster_backend(
    num_envs: int, device: str | torch.device, simulation_backend: str | None = "auto", task: str = "traffic"
) -> str:
    """Return ``"native"`` or ``"mjwarp"`` for a batch.

    ``"auto"`` steps on the native batch on CPU. On CUDA it steps small batches on the native batch and large ones
    on MuJoCo Warp; see ``mujoco_backend``.
    """

    def build(backend: str, n: int):
        return make_roadster_adapter(task, n, device, backend)

    measure = measure_adapter(build, lambda n: torch.rand(n, 4, device=device) * 2 - 1)
    path = f"roadster-sim/{task}"
    min_envs = CUDA_MIN_ENVS[task]
    return mujoco_backend(path, num_envs, device, simulation_backend, cuda_min_envs=min_envs, measure=measure)


def make_roadster_adapter(task: str, num_envs: int, device="cpu", backend: str = "native"):
    """The adapter of ``backend`` for ``num_envs`` environments of ``task``, with State on ``device``."""
    if backend == "mjwarp":
        from .warp import RoadsterWarpAdapter

        return RoadsterWarpAdapter(task, num_envs, device)
    from .batch import RoadsterNativeAdapter

    return RoadsterNativeAdapter(task, num_envs, device)
