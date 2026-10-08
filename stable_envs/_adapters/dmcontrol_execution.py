from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from . import SimulatorAdapter
from .execution import cpu_affinity, process_count

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


def make_dmc_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | DMCProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, eligible=domain in PROCESS_DOMAINS)
    if count:
        return DMCProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    args = (num_envs, device, seed)
    return local_factory(*args) if domain in ("reacher", "humanoid") else local_factory(domain, *args)
