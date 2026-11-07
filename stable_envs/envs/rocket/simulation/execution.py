"""Choose local or process execution for a Rocket batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count
from .adapter import RocketAdapter

if TYPE_CHECKING:
    import torch

    from .process import RocketProcessAdapter


def make_rocket_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    config: dict | None = None,
    frame: tuple[int, int] | None = None,
) -> RocketAdapter | RocketProcessAdapter:
    """``frame`` is the ``(size, supersample)`` of the TinyRenderer step frames the lanes draw, or None."""
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
