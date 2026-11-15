from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.mjbatch import measure_adapter, mujoco_backend
from ...._execution.policy import cpu_affinity, process_count

if TYPE_CHECKING:
    import torch

from .adapter import FetchAdapter
from .process import FetchProcessAdapter

#: ``simulation_backend`` values besides ``"auto"``: the shared MuJoCo backends, and the per-environment
#: gymnasium-robotics loop.
SIMULATION_BACKENDS = ("native", "mjwarp", "source")


def fetch_backend(
    num_envs: int,
    device: str | torch.device,
    simulation_backend: str | None = "auto",
    task: str = "PickAndPlace",
    reward_type: str = "sparse",
) -> str:
    """Return ``"native"``, ``"mjwarp"`` or ``"source"`` for a Fetch batch.

    ``"auto"`` steps on the native batch on CPU. On CUDA it steps small batches on the native batch and large ones
    on MuJoCo Warp; see ``mujoco_backend``. ``"source"`` steps each environment with gymnasium-robotics' own
    ``step`` in worker processes.
    """
    if simulation_backend == "source":
        return "source"
    if simulation_backend not in (None, "auto", *SIMULATION_BACKENDS):
        raise ValueError("simulation_backend must be 'auto', 'native', 'mjwarp' or 'source'")

    def build(backend: str, n: int):
        return make_fetch_adapter(n, device, 0, task, reward_type, simulation_backend=backend)

    measure = measure_adapter(build, 4, device)
    return mujoco_backend(f"fetch-sim/{task.lower()}", num_envs, device, simulation_backend, measure=measure)


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
