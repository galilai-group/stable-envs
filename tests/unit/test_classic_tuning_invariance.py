"""The worker-process knobs change where Acrobot lanes step, never their State or frames."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygame")
torch = pytest.importorskip("torch")

import stable_envs as snv
from stable_envs import tuning
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_worker_count_keeps_state_and_frames(workers):
    with tuning.override(workers=0):
        reference = snv.make("Acrobot", num_envs=8, seed=11, device="cpu")
    with tuning.override(workers=workers, worker_min_lanes=4):
        candidate = snv.make("Acrobot", num_envs=8, seed=11, device="cpu")
    with reference, candidate:
        assert_same_state(reference.reset(seed=11), candidate.reset(seed=11))
        rng = np.random.default_rng(2)
        actions = torch.stack([reference.action_spec.sample(8, rng=rng) for _ in range(6)], 1)
        assert_same_state(reference.step(actions, frames=slice(None)), candidate.step(actions, frames=slice(None)))
        assert_same_state(reference.reset(), candidate.reset())
