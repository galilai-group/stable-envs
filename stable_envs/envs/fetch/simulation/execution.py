from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

from .adapter import FetchAdapter
from .process import FetchProcessAdapter


def make_fetch_adapter(
    num_envs: int,
    device: str | torch.device = "cpu",
    seed: int | None = 0,
    task: str = "PickAndPlace",
    reward_type: str = "sparse",
) -> FetchAdapter:
    # Daemonic workers cannot spawn children; keep their simulators local.
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="fetch-sim")
    if count:
        return FetchProcessAdapter(
            num_envs, device, seed, task, reward_type, processes=count, threads_per_process=1, cpus=cpus
        )
    return FetchAdapter(num_envs, device, seed, task, reward_type)
