"""Choose the physics backend of a Roadster batch: native ``mujoco.batch`` on CPU or MuJoCo Warp on CUDA."""

from __future__ import annotations

import torch

from ...._execution.mjbatch import measure_adapter, mujoco_backend


def roadster_backend(
    num_envs: int, device: str | torch.device, simulation_backend: str | None = "auto", task: str = "traffic"
) -> str:
    """Return ``"native"`` or ``"mjwarp"`` for a batch.

    ``"auto"`` steps on the native batch on CPU. On CUDA it steps small batches on the native batch and large ones
    on MuJoCo Warp; see ``mujoco_backend``.
    """

    def build(backend: str, n: int):
        return make_roadster_adapter(task, n, device, backend)

    measure = measure_adapter(build, 4, device)
    return mujoco_backend(f"roadster-sim/{task}", num_envs, device, simulation_backend, measure=measure)


def make_roadster_adapter(task: str, num_envs: int, device="cpu", backend: str = "native"):
    """The adapter of ``backend`` for ``num_envs`` environments of ``task``, with State on ``device``."""
    if backend == "mjwarp":
        from .warp import RoadsterWarpAdapter

        return RoadsterWarpAdapter(task, num_envs, device)
    from .batch import RoadsterNativeAdapter

    return RoadsterNativeAdapter(task, num_envs, device)
