"""Choose the PushT physics backend: Numba on CPU, Triton on CUDA, with worker processes for large resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

    from .batch import PushTBatchAdapter

#: Backends that ``simulation_backend`` accepts besides ``"auto"``.
SIMULATION_BACKENDS = ("triton", "jax", "numba")
#: Shipped CUDA switch point: the smallest batch that steps on Triton (JAX without Triton); smaller batches step with
#: Numba. None uses the GPU backend at every batch size. The knob and the tuning cache take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
#: Triton was faster from one environment on both GPU types, because a Numba step moves the whole State to CUDA.
CUDA_MIN_ENVS: int | None = None


#: The ``select_backend`` path of the PushT physics backend choice.
PATH = "pusht-sim"


def _measure(env_type, device, config: dict):
    """Build and step ``env_type`` on one backend with ``config``, for ``stable-envs tune``."""
    from ...._execution.backends import measure_env

    options = dict(size=config["resolution"], render_action=config["render_action"], with_target=config["with_target"])
    return measure_env(lambda backend, n: env_type(num_envs=n, device=device, simulation_backend=backend, **options))


def pusht_backend(
    num_envs: int,
    device: str | torch.device,
    backend: str | None = "auto",
    *,
    discrete: bool = False,
    config: dict | None = None,
    env_type: type | None = None,
) -> str:
    """Return ``"triton"``, ``"jax"`` or ``"numba"``.

    ``"auto"`` selects Numba on CPU. On CUDA it selects Numba for small batches and Triton (JAX without Triton) for
    large ones; ``select_backend`` measures the switch point. ``stable-envs tune`` builds ``env_type`` with ``config``
    to calibrate it.
    """
    import torch

    from ...._execution.backends import Rungs, select_backend

    if backend == "triton" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    measure = None if config is None or env_type is None else _measure(env_type, device, config)
    return select_backend(
        PATH,
        num_envs,
        device,
        backend,
        {"cpu": Rungs("numba", "numba"), "cuda": Rungs("numba", _gpu_backend(), CUDA_MIN_ENVS)},
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
        if torch.device(device).type == "cpu":
            return host
        host.export_device = torch.device(device)
        return HostAdapter(host, device)
    # Workers reset the lanes; one batched call steps them.
    return adapter(num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus)
