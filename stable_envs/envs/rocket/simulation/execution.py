"""Choose the Rocket physics backend: the Numba kernel, or PyFlyt and PyBullet in this process or in workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .adapter import RocketAdapter
    from .numba_batch import RocketNumbaAdapter
    from .process import RocketProcessAdapter

#: Backends that ``simulation_backend`` accepts; ``None`` selects ``"numba"``.
SIMULATION_BACKENDS = ("numba", "pybullet")


def rocket_backend(backend: str | None = None) -> str:
    """Return ``"numba"`` or ``"pybullet"``; ``None`` selects ``"numba"`` on CPU and on CUDA."""
    if backend is None:
        return "numba"
    if backend not in SIMULATION_BACKENDS:
        raise ValueError("simulation_backend must be None, 'numba' or 'pybullet'")
    return backend


def make_rocket_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    config: dict | None = None,
    frame: tuple[int, int] | None = None,
    simulation_backend: str | None = None,
) -> RocketNumbaAdapter | RocketAdapter | RocketProcessAdapter:
    """``frame`` is the ``(size, supersample)`` of the TinyRenderer step frames, or None."""
    if rocket_backend(simulation_backend) == "numba":
        from .numba_batch import RocketNumbaAdapter

        return RocketNumbaAdapter(num_envs, device, seed, config, frame)
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
