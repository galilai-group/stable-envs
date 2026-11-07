"""Choose local or process execution for a Roadster batch."""

from __future__ import annotations

from ...._execution.policy import cpu_affinity, process_count
from .adapter import RoadsterAdapter


def make_roadster_adapter(task: str, num_envs: int, device="cpu"):
    """Return the adapter for a batch: process workers when it is large enough, else in-process.

    ``roadster-sim`` is the tuning path that sets the threshold and the worker count.
    """
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="roadster-sim") if type(num_envs) is int and num_envs > 0 else 0
    if count:
        from .process import RoadsterProcessAdapter

        # A batch that needs more workers than CPUs shares them.
        return RoadsterProcessAdapter(
            task, num_envs, device, processes=count, cpus=cpus if cpus and len(cpus) >= count else None
        )
    return RoadsterAdapter(task, num_envs, device)
