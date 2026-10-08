from __future__ import annotations

import torch

from .dmcontrol_factory import PROCESS_DOMAINS, DMCAdapter, make_local
from .dmcontrol_process import DMCProcessAdapter
from .execution import cpu_affinity, process_count


def make_dmc_adapter(domain: str, num_envs: int, device: str | torch.device, seed: int) -> DMCAdapter | DMCProcessAdapter:
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, eligible=domain in PROCESS_DOMAINS)
    if count:
        return DMCProcessAdapter(domain, num_envs, device, seed, processes=count, cpus=cpus)
    return make_local(domain, num_envs, device, seed)
