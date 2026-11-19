"""Choose the Rocket physics backend: the Numba kernel on CPU or CUDA, or PyFlyt and PyBullet."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .adapter import RocketAdapter
    from .numba_batch import RocketNumbaAdapter
    from .process import RocketProcessAdapter

_NUMBA_CUDA_FIX = (
    "Install `stable-envs[rocket-gpu]`, which adds Numba CUDA with the CUDA 13 toolkit wheels (for a torch build "
    "for CUDA 12, install `numba-cuda[cu12]` in its place), or pass simulation_backend='numba' to step on the CPU."
)
#: Backends that ``simulation_backend`` accepts besides ``"auto"``.
SIMULATION_BACKENDS = ("numba", "numba_cuda", "pybullet")


def rocket_backend(device: str | torch.device, backend: str | None = "auto") -> str:
    """Return ``"numba"``, ``"numba_cuda"`` or ``"pybullet"``.

    ``"auto"`` (or None) selects the CPU kernel on CPU and its CUDA build on CUDA. Where Numba CUDA cannot compile,
    ``"auto"`` selects the CPU kernel on CUDA and warns.
    """
    import torch

    from ...._execution.backends import select_backend

    if backend == "numba_cuda" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='numba_cuda' requires device='cuda'")
    selected = select_backend(device, backend, {"cpu": "numba", "cuda": "numba_cuda"}, choices=SIMULATION_BACKENDS)
    if selected == "numba_cuda" and backend in (None, "auto"):
        error = numba_cuda_error()
        if error is not None:
            warnings.warn(
                f"PyFlytRocketLanding steps on the CPU kernel because Numba CUDA cannot compile here ({error}). "
                f"{_NUMBA_CUDA_FIX}",
                RuntimeWarning,
                stacklevel=3,
            )
            return "numba"
    return selected


@functools.cache
def numba_cuda_error() -> str | None:
    """Return why Numba CUDA cannot compile the CUDA physics here, or None if it can.

    The compile needs libNVVM and libdevice from a CUDA toolkit, which the ``rocket-gpu`` extra installs as wheels.
    """
    try:
        from numba.cuda.cudadrv.nvvm import NVVM, LibDevice
    except ImportError as e:
        return f"{type(e).__name__}: {e}"
    try:
        NVVM()
        LibDevice()
    except Exception as e:  # noqa: BLE001 — reported with the fix
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return None


def require_numba_cuda() -> None:
    """Raise with the install steps when Numba CUDA cannot compile the CUDA physics."""
    error = numba_cuda_error()
    if error is not None:
        raise RuntimeError(
            f"simulation_backend='numba_cuda' compiles the PyFlytRocketLanding physics with Numba CUDA, which needs "
            f"libNVVM and libdevice from a CUDA toolkit, and Numba cannot load them ({error}). {_NUMBA_CUDA_FIX}"
        )


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
        require_numba_cuda()
        from .numba_cuda_batch import RocketCudaAdapter

        return RocketCudaAdapter(num_envs, device, seed, config, frame)
    from .adapter import RocketAdapter
    from .process import ENVS_PER_PROCESS

    cpus = cpu_affinity()
    count = (
        process_count(num_envs, cpus, path="rocket-sim", envs_per_process=ENVS_PER_PROCESS)
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
