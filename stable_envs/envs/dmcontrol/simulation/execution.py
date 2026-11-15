from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...._execution.backends import Rungs, measure_adapter, select_backend
from .batch import DMCBatchAdapter
from .factory import ACTION_DIMS, make_batch
from .first_family import ReacherAdapter

if TYPE_CHECKING:
    from .mjwarp import DMCWarpAdapter


#: Shipped CUDA switch points: the smallest batch that steps on MuJoCo Warp, per simulator key; smaller batches step
#: on native physics. A key that isn't here uses MuJoCo Warp at every batch size. The knob and the tuning cache
#: (``stable-envs tune``) take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
CUDA_MIN_ENVS: dict[str, int] = {
    "acrobot": 512,  # RTX 3090 512, RTX A5000 128
    "ball_in_cup": 512,  # 512, 256
    "cartpole": 512,  # 512, 128
    "cartpole-balance": 256,  # 256, 64
    "cheetah": 64,
    "finger": 128,  # 128, 64
    "finger-spin": 128,
    "finger-turn_easy": 128,  # 128, 64
    "hopper": 256,  # 256, 128
    "humanoid": 128,
    "manipulator": 512,  # 256, 512
    "pendulum": 256,
    "point_mass-easy": 64,
    "quadruped": 128,
    "reacher": 128,
    "swimmer-swimmer6": 256,
    "walker": 512,  # 256, 512
}


def _measure(key: str, device: str | torch.device):
    """Build and step ``key`` lanes on one backend, for ``stable-envs tune``."""
    return measure_adapter(
        lambda backend, n: make_dmc_adapter(key, n, device, 0, simulation_backend=backend),
        lambda n: torch.rand(n, ACTION_DIMS[key], dtype=torch.float64, device=device) * 2 - 1,
    )


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
        {"cpu": Rungs("native", "native"), "cuda": Rungs("native", "mjwarp", CUDA_MIN_ENVS.get(key))},
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
