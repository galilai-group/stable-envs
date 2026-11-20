from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...._execution.backends import select_backend
from .batch import DMCBatchAdapter
from .factory import make_batch
from .first_family import ReacherAdapter

if TYPE_CHECKING:
    from .mjwarp import DMCWarpAdapter


def dmc_backend(key: str, device: str | torch.device, backend: str | None = "auto") -> str:
    """Resolve ``backend`` to ``"native"``, ``"mjx"`` or ``"mjwarp"`` for simulator ``key``.

    ``"auto"`` uses native physics on every device. On CUDA, one copy moves its results to the device.
    """
    choices = ("native", "mjwarp", "mjx") if key == "reacher" else ("native", "mjwarp")
    if backend in ("mjx", "mjwarp") and torch.device(device).type != "cuda":
        raise ValueError(f"simulation_backend={backend!r} requires device='cuda'")
    if backend == "mjx" and key != "reacher":
        raise ValueError("DMC simulation_backend must be 'auto', 'native' or 'mjwarp'; 'mjx' is Reacher-only")
    return select_backend(device, backend, {"cpu": "native"}, choices=choices)


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    simulation_backend: str | None = "auto",
) -> DMCBatchAdapter | ReacherAdapter | DMCWarpAdapter:
    backend = dmc_backend(domain, device, simulation_backend)
    if backend == "mjwarp":
        try:
            from .mjwarp import DMCWarpAdapter
        except ImportError as exc:
            raise ImportError(
                f"simulation_backend='mjwarp' could not import its backend ({exc}): "
                "pip install 'stable-envs[dmcontrol-gpu]'"
            ) from exc

        return DMCWarpAdapter(domain, num_envs, device, seed)
    if backend == "mjx":
        from .reacher_mjx import ReacherMJXAdapter

        return ReacherMJXAdapter(num_envs, device, seed)
    return make_batch(domain, num_envs, device, seed)
