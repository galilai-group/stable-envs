"""Choose the physics backend of a Roadster batch: native ``mujoco.batch``, or MuJoCo Warp on CUDA by request."""

from __future__ import annotations

import torch

from ...._execution.mjbatch import mujoco_backend


def roadster_backend(device: str | torch.device, simulation_backend: str | None = "auto") -> str:
    """Return ``"native"`` or ``"mjwarp"`` for a batch; ``"auto"`` steps on the native batch on every device.

    See ``mujoco_backend``.
    """
    return mujoco_backend(device, simulation_backend)


def make_roadster_adapter(task: str, num_envs: int, device="cpu", backend: str = "native"):
    """The adapter of ``backend`` for ``num_envs`` environments of ``task``, with State on ``device``."""
    if backend == "mjwarp":
        from .warp import RoadsterWarpAdapter

        return RoadsterWarpAdapter(task, num_envs, device)
    from .batch import RoadsterNativeAdapter

    return RoadsterNativeAdapter(task, num_envs, device)
