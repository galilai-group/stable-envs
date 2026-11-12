"""Construct the same domain implementation in local and batch execution."""

import os
from collections.abc import Callable
from functools import partial

import torch

from .... import tuning
from ...._execution.policy import cpu_affinity
from .aligned import ALIGNED_BATCH, ALIGNED_FIELDS, MODEL_KEYS, DMCAlignedAdapter
from .aligned import CONFIG as ALIGNED
from .batch import DMCBatchAdapter
from .continuation import CONFIG as CONTINUATION
from .continuation import CONTINUATION_BATCH, CONTINUATION_FIELDS, DMCContinuationAdapter
from .first_family import FIRST_FAMILY_BATCH, FIRST_FAMILY_FIELDS, REACHER_FIELDS, FirstFamilyAdapter, ReacherAdapter
from .humanoid import HUMANOID_CONFIG, HUMANOID_FIELDS, DMCHumanoidAdapter, HumanoidBatch

DMCAdapter = FirstFamilyAdapter | ReacherAdapter | DMCAlignedAdapter | DMCContinuationAdapter | DMCHumanoidAdapter
FACTORIES: dict[str, Callable[..., DMCAdapter]] = {
    **{domain: partial(FirstFamilyAdapter, domain) for domain in FIRST_FAMILY_FIELDS},
    **{domain: partial(DMCAlignedAdapter, domain) for domain in ALIGNED},
    **{domain: partial(DMCContinuationAdapter, domain) for domain in CONTINUATION},
    "reacher": ReacherAdapter,
    "humanoid": DMCHumanoidAdapter,
}
#: Full State fields per simulator key.
FIELDS = {
    **FIRST_FAMILY_FIELDS,
    **REACHER_FIELDS,
    **ALIGNED_FIELDS,
    **CONTINUATION_FIELDS,
    "humanoid": HUMANOID_FIELDS,
}
ACTION_DIMS = {
    **dict.fromkeys(FIRST_FAMILY_FIELDS, 1),
    **dict.fromkeys(REACHER_FIELDS, 2),
    **{key: cfg["nu"] for key, cfg in ALIGNED.items()},
    **{key: cfg["nu"] for key, cfg in CONTINUATION.items()},
    "humanoid": HUMANOID_CONFIG["nu"],
}
#: Batch task per simulator key: rewards, observations and info for all lanes at once.
BATCH_TASKS = {**FIRST_FAMILY_BATCH, **ALIGNED_BATCH, **CONTINUATION_BATCH, "humanoid": HumanoidBatch}
#: State fields that follow qpos in the compact render row, in order.
RENDER_KEYS = MODEL_KEYS


def make_local(domain: str, num_envs: int, device: str | torch.device = "cpu", seed: int = 0) -> DMCAdapter:
    try:
        factory = FACTORIES[domain]
    except KeyError:
        raise ValueError(f"Unsupported DMC domain {domain!r}") from None
    return factory(num_envs, device, seed)


def batch_threads(num_envs: int) -> int:
    """The ``dmc_threads`` knob, at most one thread per CPU in the process affinity and one per lane."""
    cpus = len(cpu_affinity() or ()) or os.cpu_count() or 1
    return min(tuning.get("dmc_threads", path="dmc-sim", size=0, batch=num_envs) or 1, cpus, num_envs)


def make_batch(
    domain: str, num_envs: int, device: str | torch.device = "cpu", seed: int = 0, threads: int | None = None
) -> DMCBatchAdapter:
    """``make_local`` lanes stepped in one mujoco.batch on ``threads`` threads.

    Without ``threads``, a step uses at most ``batch_threads`` threads, and fewer when its MuJoCo work is small
    (``step_threads``).
    """
    fit = threads is None
    if fit:
        threads = batch_threads(num_envs)
    lanes = make_local(domain, num_envs, device, seed)
    return DMCBatchAdapter(lanes, BATCH_TASKS[domain], threads, fit_threads=fit)
