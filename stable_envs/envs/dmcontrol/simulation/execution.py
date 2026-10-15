from __future__ import annotations

import torch

from ...._execution.policy import cpu_affinity, process_count
from .factory import PROCESS_DOMAINS, DMCAdapter, make_local
from .process import DMCProcessAdapter


def make_dmc_adapter(
    domain: str, num_envs: int, device: str | torch.device, seed: int
) -> DMCAdapter | DMCProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, eligible=domain in PROCESS_DOMAINS, path="dmc-sim")
    if count:
        return DMCProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return make_local(domain, num_envs, device, seed)
