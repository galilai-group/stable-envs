"""Choose the Rocket physics backend: the Numba kernel on CPU or CUDA, or PyFlyt and PyBullet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .adapter import RocketAdapter
    from .numba_batch import RocketNumbaAdapter
    from .process import RocketProcessAdapter

#: Backends that ``simulation_backend`` accepts; ``None`` takes the device default.
SIMULATION_BACKENDS = ("auto", "numba", "numba_cuda", "pybullet")
#: Smallest CUDA batch that ``simulation_backend="auto"`` steps on the GPU; smaller batches step on the CPU kernel.
NUMBA_CUDA_MIN_ENVS = 1


def rocket_backend(
    num_envs: int,
    device: str | torch.device,
    backend: str | None = None,
    numba_cuda_min_envs: int = NUMBA_CUDA_MIN_ENVS,
) -> str:
    """Return ``"numba"``, ``"numba_cuda"`` or ``"pybullet"``.

    ``None`` selects the CUDA build of the kernel on CUDA and the CPU kernel on CPU. ``"auto"`` also selects the
    CPU kernel for CUDA batches smaller than ``numba_cuda_min_envs``.
    """
    import torch

    if backend is not None and backend not in SIMULATION_BACKENDS:
        raise ValueError("simulation_backend must be None, 'auto', 'numba', 'numba_cuda' or 'pybullet'")
    if isinstance(numba_cuda_min_envs, bool) or not isinstance(numba_cuda_min_envs, int) or numba_cuda_min_envs < 1:
        raise ValueError("numba_cuda_min_envs must be an int >= 1")
    cuda = torch.device(device).type == "cuda"
    if backend == "auto":
        return "numba_cuda" if cuda and num_envs >= numba_cuda_min_envs else "numba"
    if backend is None:
        return "numba_cuda" if cuda else "numba"
    if backend == "numba_cuda" and not cuda:
        raise ValueError("simulation_backend='numba_cuda' requires device='cuda'")
    return backend


def require_nvvm() -> None:
    """Raise with the install steps when Numba CUDA cannot load libNVVM, which compiles the CUDA physics."""
    try:
        from numba.cuda.cudadrv.nvvm import NVVM
    except ImportError:
        return
    try:
        NVVM()
    except Exception as e:  # noqa: BLE001 — re-raised with the fix
        raise RuntimeError(
            "PyFlytRocketLanding steps on CUDA with Numba CUDA, which compiles with libNVVM from a CUDA toolkit, "
            f"and Numba cannot load it ({type(e).__name__}: {e}). Install a CUDA toolkit as the nvdiffrast "
            "section of docs/install.md shows (for example `apt install cuda-nvcc-13-2` from NVIDIA's CUDA apt "
            "repository) and set CUDA_HOME to it, or pass simulation_backend='numba' to step on the CPU."
        ) from e


def make_rocket_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    config: dict | None = None,
    frame: tuple[int, int] | None = None,
    simulation_backend: str = "numba",
) -> RocketNumbaAdapter | RocketAdapter | RocketProcessAdapter:
    """``frame`` is the ``(size, supersample)`` of the TinyRenderer step frames, or None.

    ``simulation_backend`` is a resolved ``rocket_backend`` name.
    """
    if simulation_backend == "numba":
        from .numba_batch import RocketNumbaAdapter

        return RocketNumbaAdapter(num_envs, device, seed, config, frame)
    if simulation_backend == "numba_cuda":
        require_nvvm()
        from .numba_cuda_batch import RocketCudaAdapter

        return RocketCudaAdapter(num_envs, device, seed, config, frame)
    from .adapter import RocketAdapter
    from .process import LANES_PER_PROCESS

    cpus = cpu_affinity()
    count = (
        process_count(num_envs, cpus, path="rocket-sim", lanes_per_process=LANES_PER_PROCESS)
        if type(num_envs) is int and num_envs > 0
        else 0
    )
    if count:
        from .process import RocketProcessAdapter

        # A batch that needs more workers than CPUs shares them.
        return RocketProcessAdapter(
            num_envs,
            device,
            seed,
            config,
            processes=count,
            cpus=cpus if cpus and len(cpus) >= count else None,
            frame=frame,
        )
    return RocketAdapter(num_envs, device, seed, config, frame)
