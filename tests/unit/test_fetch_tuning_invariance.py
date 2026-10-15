"""Fetch cells give the same State for every simulator thread count."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv
from stable_envs import tuning
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("threads", [2, 3, 6])
def test_thread_count_keeps_state(threads):
    with tuning.override(workers=0, worker_min_lanes=4, fetch_threads=1):
        reference = snv.make("FetchPush", num_envs=6, seed=17, device="cpu", backend="opengl")
    with tuning.override(workers=0, worker_min_lanes=4, fetch_threads=threads):
        candidate = snv.make("FetchPush", num_envs=6, seed=17, device="cpu", backend="opengl")
    with reference, candidate:
        assert reference._simulator._workers == 1
        assert_same_state(reference.get_state(), candidate.get_state())
        actions = np.random.default_rng(31).uniform(-1, 1, (6, 5, 4))
        assert_same_state(reference.step(actions, frames=None), candidate.step(actions, frames=None))
