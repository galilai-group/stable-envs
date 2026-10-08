from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

from . import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .acrobot_process import AcrobotProcessAdapter


_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


def process_count(domain: str, num_envs: int, cpus: Sequence[int] | None) -> int:
    available = len(cpus) if cpus is not None else 1
    if (
        domain == "acrobot"
        and sys.platform == "linux"
        and num_envs >= 64
        and available >= 4
        and not mp.current_process().daemon
    ):
        return min(8, available)
    return 0


def make_classic_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | AcrobotProcessAdapter:
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = process_count(domain, num_envs, cpus)
    if count:
        from .acrobot_process import AcrobotProcessAdapter

        return AcrobotProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(domain, num_envs, device, seed)
