"""Select process execution from batch size and the allocated CPU affinity."""

import multiprocessing as mp
import os
import sys
from collections.abc import Sequence

from .. import tuning


def cpu_affinity() -> list[int] | None:
    """Return the CPUs this process may run on, or None where the platform does not report them."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def process_count(num_envs: int, cpus: Sequence[int] | None, *, eligible: bool = True, path: str | None = None) -> int:
    """Return the number of worker processes to use, or 0 to step in this process.

    Workers are used only on Linux, outside daemon processes, for at least ``worker_min_lanes`` lanes
    and 4 CPUs, up to the ``workers`` knob. ``path`` is the simulation path whose measured values apply.
    """
    available = len(cpus) if cpus is not None else 1
    if not eligible or sys.platform != "linux" or available < 4 or mp.current_process().daemon:
        return 0
    if num_envs < int(tuning.get("worker_min_lanes", path=path, size=0, batch=num_envs) or 0):
        return 0
    return min(int(tuning.get("workers", path=path, size=0, batch=num_envs) or 0), available, num_envs)
