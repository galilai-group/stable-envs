"""Row 0 of a CUDA rollout with the default backend, at several batch sizes."""

from __future__ import annotations

import numpy as np
import torch

import stable_envs as se

BATCHES = (1, 16, 128)
STEPS = 40


def row0(name: str, batch: int, seed: int = 5) -> list[tuple[str, torch.Tensor]]:
    """The leaves of row 0 of every step result: the same seed and the same row 0 actions at every batch size."""
    rng = np.random.default_rng(seed)
    with se.make(name, num_envs=max(BATCHES), device="cpu") as probe:
        actions = torch.stack([probe.action_spec.sample(max(BATCHES), rng=rng) for _ in range(STEPS)], 1)
    out = []
    with se.make(name, num_envs=batch, device="cuda") as env:
        state = env.reset(seed=seed)
        out += [(f"reset/{k}", v[:1].cpu()) for k, v in state.items(True, True)]
        for t in range(STEPS):
            result = env.step(actions[:batch, t].to("cuda"), frames=None)
            out += [(f"{t}/{k}", v[:1].cpu()) for k, v in result.items(True, True)]
    return out


def assert_batch_invariant(name: str) -> None:
    want = row0(name, BATCHES[0])
    for batch in BATCHES[1:]:
        got = row0(name, batch)
        assert [k for k, _ in got] == [k for k, _ in want], batch
        for (key, a), (_, b) in zip(want, got):
            assert a.dtype == b.dtype and torch.equal(a, b), f"{name}: {key} differs at B={batch}"
