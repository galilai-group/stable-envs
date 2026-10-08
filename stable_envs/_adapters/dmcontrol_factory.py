"""Construct the same domain implementation in local and process execution."""

from collections.abc import Callable
from functools import partial

import torch

from .dmcontrol import FIRST_FAMILY_FIELDS, FirstFamilyAdapter, ReacherAdapter
from .dmcontrol_aligned import CONFIG as ALIGNED
from .dmcontrol_aligned import DMCAlignedAdapter
from .dmcontrol_continuation import CONFIG as CONTINUATION
from .dmcontrol_continuation import DMCContinuationAdapter
from .dmcontrol_humanoid import DMCHumanoidAdapter

DMCAdapter = FirstFamilyAdapter | ReacherAdapter | DMCAlignedAdapter | DMCContinuationAdapter | DMCHumanoidAdapter
FACTORIES: dict[str, Callable[..., DMCAdapter]] = {
    **{domain: partial(FirstFamilyAdapter, domain) for domain in FIRST_FAMILY_FIELDS},
    **{domain: partial(DMCAlignedAdapter, domain) for domain in ALIGNED},
    **{domain: partial(DMCContinuationAdapter, domain) for domain in CONTINUATION},
    "reacher": ReacherAdapter,
    "humanoid": DMCHumanoidAdapter,
}
# Quadruped retains its local owner until its process route has passed the family gates.
PROCESS_DOMAINS = frozenset(FACTORIES) - {"quadruped"}


def make_local(domain: str, num_envs: int, device: str | torch.device = "cpu", seed: int = 0) -> DMCAdapter:
    try:
        factory = FACTORIES[domain]
    except KeyError:
        raise ValueError(f"Unsupported recovered DMC domain {domain!r}") from None
    return factory(num_envs, device, seed)
