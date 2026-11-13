from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.mjbatch import resolve_backend
from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

from .adapter import FetchAdapter
from .process import FetchProcessAdapter

#: Smallest CUDA batch that ``simulation_backend="auto"`` steps on MuJoCo Warp; smaller ones step on the CPU batch.
MJWARP_MIN_ENVS = 256
#: ``simulation_backend`` values: the shared MuJoCo routes, and the per-environment gymnasium-robotics loop.
SIMULATION_BACKENDS = ("default", "auto", "native", "mjwarp", "source")


def fetch_backend(
    num_envs: int, device: str | torch.device, simulation_backend: str | None = "default", mjwarp_min_envs=None
) -> str:
    """Return ``"native"``, ``"mjwarp"`` or ``"source"`` for a Fetch batch.

    ``"default"`` steps on MuJoCo Warp on CUDA and on the native batch on CPU. ``"auto"`` steps CUDA batches of
    ``mjwarp_min_envs`` or more on MuJoCo Warp and smaller ones on the native batch. ``"source"`` steps each
    environment with gymnasium-robotics' own ``step`` in worker processes.
    """
    mjwarp_min_envs = MJWARP_MIN_ENVS if mjwarp_min_envs is None else mjwarp_min_envs
    if simulation_backend == "source":
        resolve_backend(num_envs, device, "native", mjwarp_min_envs)
        return "source"
    if simulation_backend not in (None, *SIMULATION_BACKENDS):
        raise ValueError("simulation_backend must be 'default', 'auto', 'native', 'mjwarp' or 'source'")
    return resolve_backend(num_envs, device, simulation_backend, mjwarp_min_envs)


def make_fetch_adapter(
    num_envs: int,
    device: str | torch.device = "cpu",
    seed: int | None = 0,
    task: str = "PickAndPlace",
    reward_type: str = "sparse",
    simulation_backend: str = "source",
):
    if simulation_backend == "native":
        from .batch import FetchBatchAdapter

        return FetchBatchAdapter(num_envs, device, seed, task, reward_type)
    if simulation_backend == "mjwarp":
        from .mjwarp import FetchWarpAdapter

        return FetchWarpAdapter(num_envs, device, seed, task, reward_type)
    # Daemonic workers cannot spawn children; keep their simulators local.
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="fetch-sim")
    if count:
        return FetchProcessAdapter(
            num_envs, device, seed, task, reward_type, processes=count, threads_per_process=1, cpus=cpus
        )
    return FetchAdapter(num_envs, device, seed, task, reward_type)
