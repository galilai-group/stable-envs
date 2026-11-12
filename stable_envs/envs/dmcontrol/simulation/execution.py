from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .batch import DMCBatchAdapter
from .factory import make_batch
from .first_family import ReacherAdapter

if TYPE_CHECKING:
    from .mjwarp import DMCWarpAdapter


#: Smallest batch from which ``simulation_backend="auto"`` uses MuJoCo Warp on CUDA, per simulator key: the
#: smallest measured batch from which MuJoCo Warp stepped faster than native physics at every larger batch.
#: ``None`` keeps native physics at every batch.
MJWARP_MIN_ENVS: dict[str, int | None] = {
    "acrobot": 1024,
    "ball_in_cup": 256,
    "cartpole": 64,
    "cartpole-balance": 64,
    "cheetah": 1,
    "finger": 8,
    "finger-spin": 1024,
    "finger-turn_easy": 1024,
    "hopper": 256,
    "humanoid": 64,
    "manipulator": 1024,
    "pendulum": 8,
    "point_mass-easy": 8,
    "quadruped": 64,
    "reacher": 8,
    "swimmer-swimmer6": 1024,
    "walker": 1024,
}


def _check_cutoff(name: str, value: int | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1")


def dmc_backend(
    key: str,
    num_envs: int,
    device: str | torch.device,
    backend: str | None = None,
    *,
    mjwarp_min_envs: int | None = None,
) -> str:
    """Resolve ``backend`` to ``"native"``, ``"mjx"`` or ``"mjwarp"`` for simulator ``key``.

    ``None`` uses native physics on CPU and MuJoCo Warp on CUDA at every batch size.
    ``"auto"`` uses native physics below ``mjwarp_min_envs`` and MuJoCo Warp at or above it on CUDA.
    ``mjwarp_min_envs`` defaults to the measured cutoff of ``key``; a task with no cutoff stays native.
    """
    if backend not in (None, "auto", "native", "mjx", "mjwarp"):
        raise ValueError("simulation_backend must be None, 'auto', 'native', 'mjx' or 'mjwarp'")
    _check_cutoff("mjwarp_min_envs", mjwarp_min_envs, optional=True)
    cuda = torch.device(device).type == "cuda"
    if backend in ("mjx", "mjwarp") and not cuda:
        raise ValueError(f"simulation_backend={backend!r} requires device='cuda'")
    if backend == "mjx" and key != "reacher":
        raise ValueError("DMC simulation_backend must be 'auto', 'native' or 'mjwarp'; 'mjx' is Reacher-only")
    if backend not in (None, "auto"):
        return backend
    if not cuda:
        return "native"
    if backend == "auto":
        cutoff = MJWARP_MIN_ENVS.get(key) if mjwarp_min_envs is None else mjwarp_min_envs
        if cutoff is None or num_envs < cutoff:
            return "native"
    return "mjwarp"


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    simulation_backend: str | None = None,
    mjwarp_min_envs: int | None = None,
) -> DMCBatchAdapter | ReacherAdapter | DMCWarpAdapter:
    backend = dmc_backend(domain, num_envs, device, simulation_backend, mjwarp_min_envs=mjwarp_min_envs)
    if backend == "mjwarp":
        try:
            from .mjwarp import DMCWarpAdapter
        except ImportError as exc:
            raise ImportError("simulation_backend='mjwarp' requires the DMC and GPU dependencies") from exc

        return DMCWarpAdapter(domain, num_envs, device, seed)
    if backend == "mjx":
        from .reacher_mjx import ReacherMJXAdapter

        return ReacherMJXAdapter(num_envs, device, seed)
    return make_batch(domain, num_envs, device, seed)
