from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .fetch_pickandplace import FetchAdapter
from .fetch_process import FetchProcessAdapter


def make_fetch_adapter(
    num_envs: int,
    device: str | torch.device = "cpu",
    seed: int | None = 0,
    task: str = "PickAndPlace",
    reward_type: str = "sparse",
) -> FetchAdapter:
    # Daemonic workers cannot spawn children; keep their simulators local.
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    available = len(cpus) if cpus is not None else 1
    if sys.platform == "linux" and num_envs >= 64 and available >= 4 and not mp.current_process().daemon:
        return FetchProcessAdapter(
            num_envs, device, seed, task, reward_type, processes=min(8, available), threads_per_process=1, cpus=cpus
        )
    return FetchAdapter(num_envs, device, seed, task, reward_type)
