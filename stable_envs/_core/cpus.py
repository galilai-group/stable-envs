"""The number of CPUs this process can use: its affinity, capped by a cgroup CPU quota.

Containers often see every host CPU in their affinity but get a CPU quota (cgroup v2 ``cpu.max``,
cgroup v1 ``cpu.cfs_quota_us``) of a few CPUs. Thread pools sized from the affinity then run many
more threads than the quota allows, and the kernel throttles the whole process.
"""

from __future__ import annotations

import functools
import math
import os
import sys
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


def _v2_quota(text: str) -> float | None:
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    quota, period = int(parts[0]), int(parts[1])
    return quota / period if quota > 0 and period > 0 else None


def _v1_quota(directory: Path) -> float | None:
    try:
        quota = int((directory / "cpu.cfs_quota_us").read_text())
        period = int((directory / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return quota / period if quota > 0 and period > 0 else None


def _cgroup_dirs(root: Path, proc_cgroup: Path) -> list[Path]:
    """The process's cgroup directories and their ancestors, v2 and v1 CPU controllers, deepest first."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        lines = []
    dirs = []
    for line in lines:
        if line.count(":") < 2:
            continue
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            base = root
        elif "cpu" in controllers.split(","):
            base = next(
                (root / name for name in ("cpu", "cpu,cpuacct", "cpuacct,cpu") if (root / name).is_dir()), None
            )
            if base is None:
                continue
        else:
            continue
        rel = Path(path.lstrip("/"))
        for depth in range(len(rel.parts), -1, -1):
            dirs.append(base.joinpath(*rel.parts[:depth]))
    for fallback in (root, root / "cpu", root / "cpu,cpuacct"):
        if fallback not in dirs:
            dirs.append(fallback)
    return dirs


def cgroup_cpu_quota(root: Path = CGROUP_ROOT, proc_cgroup: Path = Path("/proc/self/cgroup")) -> float | None:
    """The smallest CPU quota, in CPUs, of the process's cgroup and its ancestors, or None without one."""
    quotas = []
    for directory in _cgroup_dirs(root, proc_cgroup):
        try:
            quota = _v2_quota((directory / "cpu.max").read_text())
        except (OSError, ValueError):
            quota = _v1_quota(directory)
        if quota is not None:
            quotas.append(quota)
    return min(quotas) if quotas else None


def _affinity() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@functools.cache
def cpu_limit() -> int:
    """CPUs this process can use at once: the affinity, capped by ``ceil`` of any cgroup CPU quota."""
    available = _affinity()
    quota = cgroup_cpu_quota() if sys.platform.startswith("linux") else None
    if quota is not None:
        available = min(available, math.ceil(quota))
    return max(1, available)


def limit_thread_defaults() -> None:
    """Lower the default Numba and Torch thread counts to ``cpu_limit()``.

    A thread count that the user sets (``NUMBA_NUM_THREADS``, ``OMP_NUM_THREADS``, ``torch.set_num_threads``
    before this call) is kept: only counts above the limit that no variable sets are lowered.
    """
    limit = cpu_limit()
    if "NUMBA_NUM_THREADS" not in os.environ:
        if "numba" in sys.modules:
            numba_threads()
        elif limit < (os.cpu_count() or limit):
            # Numba sizes its pool from this variable when it is first imported.
            os.environ["NUMBA_NUM_THREADS"] = str(limit)
    if "OMP_NUM_THREADS" not in os.environ and "torch" in sys.modules:
        import torch

        if torch.get_num_threads() > limit:
            torch.set_num_threads(limit)


def numba_threads() -> int:
    """Numba's thread count for this thread, first lowered to ``cpu_limit()`` unless ``NUMBA_NUM_THREADS`` is set."""
    import numba

    if "NUMBA_NUM_THREADS" not in os.environ and numba.get_num_threads() > cpu_limit():
        numba.set_num_threads(cpu_limit())
    return numba.get_num_threads()
