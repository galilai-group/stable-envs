"""Measured DMC routing candidate; public activation requires the default gate.

The current constructors remain local. The immutable Oscar default-gate bundle
activates this factory without runtime flags; accepted activation can then use
those exact constructor changes. Quadruped is excluded pending its frame gate.
"""

import multiprocessing as mp
import os
import sys

from .dmcontrol_process import DMCProcessAdapter

PROCESS_DOMAINS = frozenset(("acrobot", "pendulum", "cartpole", "reacher", "cheetah", "hopper",
                             "walker", "ball_in_cup", "finger", "manipulator", "humanoid"))


def process_count(domain, num_envs, cpus):
    available = len(cpus) if cpus is not None else 1
    if (domain in PROCESS_DOMAINS and sys.platform == "linux" and num_envs >= 64
            and available >= 4 and not mp.current_process().daemon):
        return min(8, available)
    return 0


def make_dmc_adapter(domain, num_envs, device, seed, local_factory):
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = process_count(domain, num_envs, cpus)
    if count:
        return DMCProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    args = (num_envs, device, seed)
    return local_factory(*args) if domain in ("reacher", "humanoid") else local_factory(domain, *args)
