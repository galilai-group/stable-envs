"""Private Fetch execution selection; no scheduling options in the public API."""

import multiprocessing as mp
import os
import sys

from .fetch_pickandplace import FetchAdapter
from .fetch_process import FetchProcessAdapter


def make_fetch_adapter(num_envs, device="cpu", seed=0, task="PickAndPlace", reward_type="sparse"):
    # The measured process strategy targets large Linux batches. Preserve the
    # existing local path on other platforms and where child processes cannot
    # be created (e.g. a daemon DataLoader worker).
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    available = len(cpus) if cpus is not None else 1
    if sys.platform == "linux" and num_envs >= 64 and available >= 4 and not mp.current_process().daemon:
        return FetchProcessAdapter(
            num_envs, device, seed, task, reward_type, processes=min(8, available), threads_per_process=1, cpus=cpus
        )
    return FetchAdapter(num_envs, device, seed, task, reward_type)
