"""Select process execution from batch size and the allocated CPU affinity."""

import multiprocessing as mp
import os
import sys
from collections.abc import Sequence

from .. import tuning
from .._core.cpus import cpu_limit


def cpu_affinity() -> list[int] | None:
    """Return the CPUs this process may run on, or None where the platform does not report them."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def process_count(
    num_envs: int,
    cpus: Sequence[int] | None,
    *,
    eligible: bool = True,
    path: str | None = None,
    envs_per_process: int | None = None,
    min_envs: int | None = None,
) -> int:
    """Return the number of worker processes to use, or 0 to step in this process.

    Workers are used only on Linux, outside daemon processes, for at least ``worker_min_envs`` environments
    and 4 CPUs, up to the ``workers`` setting. ``path`` is the simulation path whose measured values apply.
    ``min_envs`` is the path's own ``worker_min_envs`` default. A batch larger than ``envs_per_process``
    uses at least enough workers to hold it on any platform.
    """
    available = min(len(cpus), cpu_limit()) if cpus is not None else 1
    if not eligible or mp.current_process().daemon:
        return 0
    needed = -(-num_envs // envs_per_process) if envs_per_process and num_envs > envs_per_process else 0
    if sys.platform != "linux" or available < 4:
        return needed
    default = {} if min_envs is None else {"default": min_envs}
    if num_envs < int(tuning.get("worker_min_envs", path=path, size=0, batch=num_envs, **default) or 0):
        return needed
    return max(needed, min(int(tuning.get("workers", path=path, size=0, batch=num_envs) or 0), available, num_envs))
