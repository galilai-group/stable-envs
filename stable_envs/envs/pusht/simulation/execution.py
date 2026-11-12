"""Choose the PushT physics backend: Numba on CPU, Triton on CUDA, with worker processes for large resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .batch import PushTBatchAdapter

#: Backends that ``simulation_backend`` accepts; ``None`` takes the device default.
SIMULATION_BACKENDS = ("auto", "triton", "jax", "numba")
#: Smallest CUDA batch that ``simulation_backend="auto"`` steps on the GPU; smaller batches step with Numba.
TRITON_MIN_ENVS = 1


def pusht_backend(
    num_envs: int, device: str | torch.device, backend: str | None = None, triton_min_envs: int = TRITON_MIN_ENVS
) -> str:
    """Return ``"triton"``, ``"jax"`` or ``"numba"``.

    ``None`` selects Triton on CUDA (JAX without Triton) and Numba on CPU. ``"auto"`` also selects Numba
    for CUDA batches smaller than ``triton_min_envs``.
    """
    import torch

    if backend is not None and backend not in SIMULATION_BACKENDS:
        raise ValueError("simulation_backend must be None, 'auto', 'triton', 'jax' or 'numba'")
    if isinstance(triton_min_envs, bool) or not isinstance(triton_min_envs, int) or triton_min_envs < 1:
        raise ValueError("triton_min_envs must be an int >= 1")
    cuda = torch.device(device).type == "cuda"
    if backend == "auto":
        backend = None if cuda and num_envs >= triton_min_envs else "numba"
    if backend is None:
        return _gpu_backend() if cuda else "numba"
    if backend == "triton" and not cuda:
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    return backend


def _gpu_backend() -> str:
    from .. import raster

    return "triton" if raster.HAVE_TRITON else "jax"


def make_pusht_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    *,
    discrete: bool,
    config: dict,
    simulation_backend: str | None = None,
    triton_min_envs: int = TRITON_MIN_ENVS,
) -> PushTBatchAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="pusht-sim") if type(num_envs) is int and num_envs > 0 else 0
    backend = pusht_backend(num_envs, device, simulation_backend, triton_min_envs)
    if backend == "triton":
        from .triton_batch import PushTTritonAdapter as adapter
    elif backend == "jax":
        from .jax_batch import PushTJaxAdapter as adapter
    else:
        from .numba_batch import PushTNumbaAdapter as adapter
    # Workers reset the lanes; one batched call steps them.
    return adapter(num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus)
