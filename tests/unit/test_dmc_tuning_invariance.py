"""DMC cells give the same frames and State for every tuning knob value."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dm_control")
torch = pytest.importorskip("torch")

import stable_envs as snv
from stable_envs import tuning
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit


def _moved(env, steps=3):
    rng = np.random.default_rng(5)
    actions = torch.stack([env.action_spec.sample(env.num_envs, rng=rng) for _ in range(steps)], 1)
    return env.step(actions, frames=None)["state"]


@pytest.mark.gl
def test_cpu3d_chunk_keeps_trackcom_frames():
    with snv.make("DMCCheetahRun", num_envs=5, device="cpu") as env:
        state = _moved(env)
        reference = env.render(state)
        for chunk in (1, 2, 3, 16, 64):
            with tuning.override(cpu3d_chunk=chunk):
                assert torch.equal(env.render(state), reference), chunk


@pytest.mark.gl
@pytest.mark.parametrize("supersample", [1, 2])
def test_cpu_ray_budget_keeps_fixed_camera_frames(supersample):
    with snv.make("DMCReacher", num_envs=5, size=64, supersample=supersample, device="cpu") as env:
        state = _moved(env)
        reference = env.render(state)
        for budget in (1 << 12, 1 << 13, 1 << 15, 1 << 21, 1 << 23):
            with tuning.override(dmc_cpu_ray_budget=budget):
                assert torch.equal(env.render(state), reference), budget


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_worker_count_keeps_state(workers):
    with tuning.override(workers=0):
        reference = snv.make("DMCCartpoleSwingup", num_envs=6, seed=3, device="cpu")
    with tuning.override(workers=workers, worker_min_lanes=4):
        candidate = snv.make("DMCCartpoleSwingup", num_envs=6, seed=3, device="cpu")
    with reference, candidate:
        assert_same_state(reference.reset(seed=3), candidate.reset(seed=3))
        rng = np.random.default_rng(9)
        actions = torch.stack([reference.action_spec.sample(6, rng=rng) for _ in range(4)], 1)
        assert_same_state(reference.step(actions, frames=None), candidate.step(actions, frames=None))
