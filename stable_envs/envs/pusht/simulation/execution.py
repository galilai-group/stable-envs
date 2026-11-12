"""Choose CPU execution or batched JAX CUDA physics for PushT, with worker processes for large batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._execution.policy import cpu_affinity, process_count
from .replay_batch import PushTReplayAdapter

if TYPE_CHECKING:
    import torch

    from .process import PushTProcessAdapter


def make_pusht_adapter(
    num_envs: int, device: str | torch.device, seed: int | None, *, discrete: bool, config: dict
) -> PushTReplayAdapter | PushTProcessAdapter:
    import torch

    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="pusht-sim") if type(num_envs) is int and num_envs > 0 else 0
    if torch.device(device).type == "cuda":
        from .jax_batch import PushTJaxAdapter

        # Workers reset the lanes; JAX steps them on the GPU.
        return PushTJaxAdapter(num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus)
    if count:
        from .process import PushTProcessAdapter

        return PushTProcessAdapter(
            num_envs, device, seed, discrete=discrete, config=config, processes=count, cpus=cpus
        )
    return PushTReplayAdapter(num_envs, device, seed, discrete=discrete, config=config)
