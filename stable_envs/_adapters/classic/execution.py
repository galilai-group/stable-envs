from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ..._execution.policy import cpu_affinity, process_count
from ..._execution.protocol import SimulatorAdapter

if TYPE_CHECKING:
    import torch

    from .process import AcrobotProcessAdapter


_AdapterT = TypeVar("_AdapterT", bound=SimulatorAdapter)


def make_classic_adapter(
    domain: str,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    local_factory: Callable[..., _AdapterT],
) -> _AdapterT | AcrobotProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, eligible=domain == "acrobot")
    if count:
        from .process import AcrobotProcessAdapter

        return AcrobotProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(domain, num_envs, device, seed)
