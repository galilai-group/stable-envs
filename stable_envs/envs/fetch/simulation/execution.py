from __future__ import annotations

import torch

from ...._execution.backends import measure_env
from ...._execution.mjbatch import mujoco_backend
from ...._execution.policy import cpu_affinity, process_count
from .adapter import FetchAdapter
from .process import FetchProcessAdapter

#: Shipped CUDA switch points: the smallest batch that steps on MuJoCo Warp; smaller batches step on ``mujoco.batch``.
#: The knob and the tuning cache (``stable-envs tune``) take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
CUDA_MIN_ENVS: dict[str, int | None] = {
    "Reach": 256,
    "Push": 256,  # RTX 3090 128, RTX A5000 256
    "Slide": 256,
    "PickAndPlace": 256,
}
#: ``simulation_backend`` values besides ``"auto"``: the shared MuJoCo backends, and the per-environment
#: gymnasium-robotics loop.
SIMULATION_BACKENDS = ("native", "mjwarp", "source")


def fetch_path(task: str) -> str:
    """The simulation path of a Fetch task, which keys its switch point in the tuning cache."""
    return f"fetch-sim/{task.lower()}"


def fetch_backend(
    num_envs: int,
    device: str | torch.device,
    simulation_backend: str | None = "auto",
    task: str = "PickAndPlace",
    reward_type: str = "sparse",
    env_type: type | None = None,
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

    measure = (
        None
        if env_type is None
        else measure_env(lambda backend, n: env_type(num_envs=n, device=device, simulation_backend=backend))
    )
    path = fetch_path(task)
    min_envs = CUDA_MIN_ENVS[task]
    return mujoco_backend(path, num_envs, device, simulation_backend, cuda_min_envs=min_envs, measure=measure)


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
