"""Bounded full-State semantics gate against the pinned stable-worldmodel Reacher wrapper."""

import os

import pytest
import torch

import stable_envs as se

pytest.importorskip("dm_control")
pytestmark = pytest.mark.swm

from stable_envs._core.state_schema import get_field
from tests.helpers.reference.gates.reacher import run_gate


def test_reacher_full_state_semantics():
    with se.make("DMCReacher", device="cpu", num_envs=1) as env:
        assert env.action_spec.dtype == torch.float64
        state = env.get_state()
        assert get_field(state, "rng/keys").dtype == torch.int64
        assert get_field(state, "lifecycle/reset_next_step").dtype == torch.bool
    result = run_gate(os.environ.get("SE_REACHER_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])
