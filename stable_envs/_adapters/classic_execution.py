"""Measured Acrobot selection; public activation is gated separately."""

import multiprocessing as mp
import os
import sys


def process_count(domain, num_envs, cpus):
    available = len(cpus) if cpus is not None else 1
    if (domain == "acrobot" and sys.platform == "linux" and num_envs >= 64
            and available >= 4 and not mp.current_process().daemon):
        return min(8, available)
    return 0


def make_classic_adapter(domain, num_envs, device, seed, local_factory):
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = process_count(domain, num_envs, cpus)
    if count:
        from .acrobot_process import AcrobotProcessAdapter

        return AcrobotProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(domain, num_envs, device, seed)
