from __future__ import annotations

import torch

from .batch import DMCBatchAdapter
from .factory import make_batch
from .first_family import ReacherAdapter


def reacher_backend(num_envs: int, device: str | torch.device, backend: str = "auto", mjx_min_envs: int = 32) -> str:
    """Use native physics for small batches and MJX for larger CUDA batches."""
    if backend not in ("auto", "native", "mjx"):
        raise ValueError("simulation_backend must be 'auto', 'native' or 'mjx'")
    if isinstance(mjx_min_envs, bool) or not isinstance(mjx_min_envs, int) or mjx_min_envs < 1:
        raise ValueError("mjx_min_envs must be an int >= 1")
    cuda = torch.device(device).type == "cuda"
    if backend == "mjx" and not cuda:
        raise ValueError("simulation_backend='mjx' requires device='cuda'")
    if backend == "auto":
        return "mjx" if cuda and num_envs >= mjx_min_envs else "native"
    return backend


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    *,
    simulation_backend: str = "auto",
    mjx_min_envs: int = 32,
) -> DMCBatchAdapter | ReacherAdapter:
    if domain == "reacher" and reacher_backend(num_envs, device, simulation_backend, mjx_min_envs) == "mjx":
        from .reacher_mjx import ReacherMJXAdapter

        return ReacherMJXAdapter(num_envs, device, seed)
    return make_batch(domain, num_envs, device, seed)
