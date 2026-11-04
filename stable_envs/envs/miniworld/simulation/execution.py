"""Choose local or process execution for a MovingBlockWorld batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count
from .adapter import MovingBlockWorldAdapter

if TYPE_CHECKING:
    import torch

    from .process import MiniWorldProcessAdapter


def make_miniworld_adapter(
    config: dict, num_envs: int, size: int, device: str | torch.device = "cpu", seed: int | None = 0
) -> MovingBlockWorldAdapter | MiniWorldProcessAdapter:
    # Daemonic workers cannot spawn children; keep their simulators local.
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="miniworld-sim") if type(num_envs) is int and num_envs > 0 else 0
    if count:
        from .process import MiniWorldProcessAdapter

        return MiniWorldProcessAdapter(
            config, num_envs, size, device, seed, processes=count, cpus=cpus if cpus and len(cpus) >= count else None
        )
    return MovingBlockWorldAdapter(config, num_envs, size, device)
