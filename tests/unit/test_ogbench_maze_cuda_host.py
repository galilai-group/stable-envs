"""OGBench maze steps on CUDA keep the bookkeeping on the host and give the same outputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as se
from stable_envs._execution.process import ProcessAdapter
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("frames", [None, slice(None)])
def test_host_bookkeeping_matches_device_bookkeeping(monkeypatch, frames):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    actions = torch.as_tensor(np.random.default_rng(3).uniform(-1, 1, (64, 6, 2)), dtype=torch.float32, device="cuda")
    outputs = []
    for host in (True, False):
        monkeypatch.setattr(ProcessAdapter, "host_outputs", host)
        with se.make("PointMazeLargeNavigate", num_envs=64, device="cuda", seed=5) as env:
            env.reset(seed=5)
            state = env.get_state()
            state["root", "position"][0] = state["task", "goal_pos"][0, :2]
            env.set_state(state)
            outputs.append([env.step(actions[:, :1], frames=frames), env.step(actions[:, 1:], frames=frames)])
    for got, want in zip(outputs[0], outputs[1], strict=True):
        assert got.device == want.device
        assert_same_state(got, want)
