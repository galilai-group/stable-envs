"""Choose the physics backend of a Roadster batch: native ``mujoco.batch`` on CPU or MuJoCo Warp on CUDA."""

from __future__ import annotations

import torch

from ...._execution.mjbatch import resolve_backend

#: Smallest CUDA batch that ``simulation_backend="auto"`` steps on MuJoCo Warp; smaller ones step on the CPU batch.
MJWARP_MIN_ENVS = 64
SIMULATION_BACKENDS = ("default", "auto", "native", "mjwarp")


def roadster_backend(
    num_envs: int, device: str | torch.device, simulation_backend: str | None = "default", mjwarp_min_envs=None
) -> str:
    """Return ``"native"`` or ``"mjwarp"`` for a batch.

    ``"default"`` steps on MuJoCo Warp on CUDA and on the native batch on CPU. ``"auto"`` steps CUDA batches of
    ``mjwarp_min_envs`` or more on MuJoCo Warp and smaller ones on the native batch.
    """
    if simulation_backend not in (None, *SIMULATION_BACKENDS):
        raise ValueError("simulation_backend must be 'default', 'auto', 'native' or 'mjwarp'")
    min_envs = MJWARP_MIN_ENVS if mjwarp_min_envs is None else mjwarp_min_envs
    return resolve_backend(num_envs, device, simulation_backend, min_envs)


def make_roadster_adapter(task: str, num_envs: int, device="cpu", backend: str = "native"):
    """The adapter of ``backend`` for ``num_envs`` environments of ``task``, with State on ``device``."""
    if backend == "mjwarp":
        from .warp import RoadsterWarpAdapter

        return RoadsterWarpAdapter(task, num_envs, device)
    from .batch import RoadsterNativeAdapter

    return RoadsterNativeAdapter(task, num_envs, device)
