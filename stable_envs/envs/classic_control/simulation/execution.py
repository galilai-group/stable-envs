from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ...._execution.policy import cpu_affinity, process_count
from ...._execution.protocol import SimulatorAdapter

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
    count = process_count(num_envs, cpus, eligible=domain == "acrobot", path="classic-sim")
    if count:
        from .process import AcrobotProcessAdapter

        return AcrobotProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(domain, num_envs, device, seed)


def make_step_adapter(domain: str, backend: str, num_envs: int, device: str | torch.device, seed: int, sink=None):
    """The simulator of ``backend``; on an accelerator, Gymnasium and NumPy lanes in this process step on the CPU.

    Their results move to ``device`` in one copy. ``sink`` is the environment's rows that the CUDA Torch simulator
    writes (see ``DeviceClassicAdapter``).
    """
    import torch

    from ...._execution.host import HostAdapter

    device = torch.device(device)
    if backend == "torch" and device.type == "cuda":
        from .device import DeviceClassicAdapter

        return DeviceClassicAdapter(domain, num_envs, device, seed, sink)
    where = device if backend == "torch" else torch.device("cpu")
    if backend == "gym":
        from .local import ClassicControlAdapter

        # Worker processes move their outputs to ``device`` themselves.
        adapter = make_classic_adapter(
            domain, num_envs, device, seed, lambda d, n, _, s: ClassicControlAdapter(d, n, where, s)
        )
    else:
        from .batched import BatchedClassicAdapter

        adapter = BatchedClassicAdapter(domain, num_envs, where, seed, backend)
    return adapter if adapter.device == device else HostAdapter(adapter, device)
