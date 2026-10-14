"""Select process execution from batch size and the allocated CPU affinity."""

import multiprocessing as mp
import os
import sys
from collections.abc import Sequence


def cpu_affinity() -> list[int] | None:
    """Return the CPUs this process may run on, or None where the platform does not report them."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def process_count(num_envs: int, cpus: Sequence[int] | None, *, eligible: bool = True) -> int:
    """Return the number of worker processes to use, or 0 to step in this process.

    Workers are used only on Linux, outside daemon processes, for at least 64 lanes and 4 CPUs.
    """
    available = len(cpus) if cpus is not None else 1
    if eligible and sys.platform == "linux" and num_envs >= 64 and available >= 4 and not mp.current_process().daemon:
        return min(8, available)
    return 0
