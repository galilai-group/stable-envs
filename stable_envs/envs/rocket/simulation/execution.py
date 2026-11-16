"""Choose the Rocket physics backend: the Numba kernel on CPU or CUDA, or PyFlyt and PyBullet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .adapter import RocketAdapter
    from .numba_batch import RocketNumbaAdapter
    from .process import RocketProcessAdapter

#: Backends that ``simulation_backend`` accepts besides ``"auto"``.
SIMULATION_BACKENDS = ("numba", "numba_cuda", "pybullet")
#: Shipped CUDA switch point of ``"auto"``: the smallest batch that steps on ``"numba_cuda"``; smaller batches step on
#: the CPU kernel. None uses ``"numba_cuda"`` at every batch size. The setting and the tuning cache take precedence.
#: Measured on RTX 3090 and RTX A5000 hosts with ``python -m benchmarks.backend_switch`` (docs/performance-tuning.md).
CUDA_MIN_ENVS: int | None = None


def _measure(device, config: dict | None):
    """Build and step Rocket lanes on one backend, for ``stable-envs tune``."""
    import torch

    from ...._execution.backends import measure_adapter

    return measure_adapter(
        lambda backend, n: make_rocket_adapter(n, device, 0, config, None, backend),
        lambda n: torch.zeros((n, 7), dtype=torch.float64, device=device),
    )


def rocket_backend(
    num_envs: int, device: str | torch.device, backend: str | None = "auto", config: dict | None = None
) -> str:
    """Return ``"numba"``, ``"numba_cuda"`` or ``"pybullet"``.

    ``"auto"`` (or None) selects the CPU kernel on CPU. On CUDA it selects the CPU kernel below ``CUDA_MIN_ENVS`` and
    the CUDA build of the kernel from it. ``stable-envs tune`` measures with the simulator configuration ``config``.
    """
    import torch

    from ...._execution.backends import BackendPair, select_backend

    if backend == "numba_cuda" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='numba_cuda' requires device='cuda'")
    return select_backend(
        "rocket-sim/landing",
        num_envs,
        device,
        backend,
        {"cpu": BackendPair("numba", "numba"), "cuda": BackendPair("numba", "numba_cuda", CUDA_MIN_ENVS)},
        choices=SIMULATION_BACKENDS,
        measure=_measure(device, config) if type(num_envs) is int and num_envs > 0 else None,
    )


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
