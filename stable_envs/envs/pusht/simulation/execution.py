"""Choose the PushT physics backend: Numba on CPU, Triton on CUDA, with worker processes for large resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .batch import PushTBatchAdapter

#: Backends that ``simulation_backend`` accepts besides ``"auto"``.
SIMULATION_BACKENDS = ("triton", "jax", "numba")


def _measure(device, discrete: bool, config: dict):
    """Build and step PushT lanes on one backend, for the backend calibration."""
    import torch

    def measure(backend: str, num_envs: int):
        adapter = make_pusht_adapter(
            num_envs, device, 0, discrete=discrete, config=config, simulation_backend=backend
        )
        adapter.reset(0)
        if discrete:
            actions = torch.randint(0, 16, (num_envs, 2), device=device)
        else:
            actions = torch.rand(num_envs, 2, device=device) * 2 - 1
        return (lambda: adapter.step(actions)), adapter.close

    return measure


def pusht_backend(
    num_envs: int,
    device: str | torch.device,
    backend: str | None = "auto",
    *,
    discrete: bool = False,
    config: dict | None = None,
) -> str:
    """Return ``"triton"``, ``"jax"`` or ``"numba"``.

    ``"auto"`` selects Numba on CPU. On CUDA it selects Numba for small batches and Triton (JAX without Triton) for
    large ones; ``select_backend`` measures the switch point. ``config`` is the adapter configuration that the
    calibration builds.
    """
    import torch

    from ...._execution.backends import select_backend

    if backend == "triton" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    measure = None if config is None else _measure(device, discrete, config)
    return select_backend(
        "pusht-sim",
        num_envs,
        device,
        backend,
        {"cpu": ("numba", "numba"), "cuda": ("numba", _gpu_backend())},
        choices=SIMULATION_BACKENDS,
        measure=measure,
    )


def _gpu_backend() -> str:
    from .. import raster

    return "triton" if raster.HAVE_TRITON else "jax"


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
    count = process_count(num_envs, cpus, path="pusht-sim") if type(num_envs) is int and num_envs > 0 else 0
    backend = pusht_backend(num_envs, device, simulation_backend, discrete=discrete, config=config)
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
        return host if torch.device(device).type == "cpu" else HostAdapter(host, device)
    # Workers reset the lanes; one batched call steps them.
    return adapter(num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus)
