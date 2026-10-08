from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

from . import SimulatorAdapter

if TYPE_CHECKING:
    import torch

from .dmcontrol_process import DMCProcessAdapter

PROCESS_DOMAINS = frozenset(
    (
        "acrobot",
        "pendulum",
        "cartpole",
        "reacher",
        "cheetah",
        "hopper",
        "walker",
        "ball_in_cup",
        "finger",
        "manipulator",
        "humanoid",
    )
)


_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


def process_count(domain: str, num_envs: int, cpus: Sequence[int] | None) -> int:
    available = len(cpus) if cpus is not None else 1
    if (
        domain in PROCESS_DOMAINS
        and sys.platform == "linux"
        and num_envs >= 64
        and available >= 4
        and not mp.current_process().daemon
    ):
        return min(8, available)
    return 0


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | DMCProcessAdapter:
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = process_count(domain, num_envs, cpus)
    if count:
        return DMCProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    args = (num_envs, device, seed)
    return local_factory(*args) if domain in ("reacher", "humanoid") else local_factory(domain, *args)
