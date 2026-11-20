"""Choose the PushT physics backend (Numba by default, Triton or JAX by request) and the reset workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.backends import select_backend
from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .batch import PushTBatchAdapter

#: Backends that ``simulation_backend`` accepts besides ``"auto"``.
SIMULATION_BACKENDS = ("triton", "jax", "numba")
#: The path of the PushT physics settings.
PATH = "pusht-sim"
#: Smallest batch whose resets run in worker processes, unless the ``worker_min_envs`` setting is set. An environment's source
#: reset takes milliseconds of Python. From this batch size, parallel resets save more time than the round trip to
#: the workers costs. docs/performance-tuning.md (Worker processes) says how the value was measured.
RESET_WORKER_MIN_ENVS = 8


def pusht_backend(device: str | torch.device, backend: str | None = "auto") -> str:
    """Return ``"triton"``, ``"jax"`` or ``"numba"``.

    ``"auto"`` selects Numba on every device. On CUDA, Numba steps on the CPU and one copy moves the results to the
    device. ``"triton"`` (CUDA only) and ``"jax"`` select a GPU backend explicitly.
    """
    import torch

    if backend == "triton" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    return select_backend(device, backend, {"cpu": "numba"}, choices=SIMULATION_BACKENDS)


def make_pusht_adapter(
    num_envs: int,
    device: str | torch.device,
    seed: int | None,
    *,
    discrete: bool,
    config: dict,
    simulation_backend: str | None = "auto",
) -> PushTBatchAdapter:
    cpus = cpu_affinity()
    valid = type(num_envs) is int and num_envs > 0
    count = process_count(num_envs, cpus, path=PATH, min_envs=RESET_WORKER_MIN_ENVS) if valid else 0
    backend = pusht_backend(device, simulation_backend)
    if backend == "triton":
        from .triton_batch import PushTTritonAdapter as adapter
    elif backend == "jax":
        from .jax_batch import PushTJaxAdapter as adapter
    else:
        import torch

        from ...._execution.host import HostAdapter
        from .numba_batch import PushTNumbaAdapter

        # Numba steps the host State; only the results move to an accelerator.
        host = PushTNumbaAdapter(num_envs, "cpu", seed, discrete=discrete, config=config, processes=count, cpus=cpus)
        if torch.device(device).type == "cpu":
            return host
        host.export_device = torch.device(device)
        return HostAdapter(host, device)
    # Workers reset the environments; one batched call steps them.
    return adapter(num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus)
