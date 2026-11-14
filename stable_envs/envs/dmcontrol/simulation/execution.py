from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...._execution.backends import select_backend
from .batch import DMCBatchAdapter
from .factory import ACTION_DIMS, make_batch
from .first_family import ReacherAdapter

if TYPE_CHECKING:
    from .mjwarp import DMCWarpAdapter


def _measure(key: str, device: str | torch.device):
    """Build and step ``key`` lanes on one backend, for the backend calibration."""

    def measure(backend: str, num_envs: int):
        adapter = make_dmc_adapter(key, num_envs, device, 0, simulation_backend=backend)
        adapter.reset(0)
        actions = torch.rand(num_envs, ACTION_DIMS[key], dtype=torch.float64, device=device) * 2 - 1
        return (lambda: adapter.step(actions)), adapter.close

    return measure


def dmc_backend(key: str, num_envs: int, device: str | torch.device, backend: str | None = "auto") -> str:
    """Resolve ``backend`` to ``"native"``, ``"mjx"`` or ``"mjwarp"`` for simulator ``key``.

    ``"auto"`` uses native physics on CPU. On CUDA it uses native physics for small batches and MuJoCo Warp for large
    ones; ``select_backend`` measures the switch point.
    """
    choices = ("native", "mjwarp", "mjx") if key == "reacher" else ("native", "mjwarp")
    if backend in ("mjx", "mjwarp") and torch.device(device).type != "cuda":
        raise ValueError(f"simulation_backend={backend!r} requires device='cuda'")
    if backend == "mjx" and key != "reacher":
        raise ValueError("DMC simulation_backend must be 'auto', 'native' or 'mjwarp'; 'mjx' is Reacher-only")
    return select_backend(
        f"dmc-sim/{key}",
        num_envs,
        device,
        backend,
        {"cpu": ("native", "native"), "cuda": ("native", "mjwarp")},
        choices=choices,
        measure=_measure(key, device),
    )


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    simulation_backend: str | None = "auto",
) -> DMCBatchAdapter | ReacherAdapter | DMCWarpAdapter:
    backend = dmc_backend(domain, num_envs, device, simulation_backend)
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
