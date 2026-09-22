"""Bounded full-State recovery gate against the pinned SWM Reacher wrapper."""
import os

import pytest
import torch

import stable_envs as se

pytest.importorskip('dm_control')
pytest.importorskip('stable_worldmodel')

from scripts.recovery.reacher_gate import run_gate


def test_reacher_full_state_semantics():
    with se.make("DMCReacherHard", device="cpu", num_envs=1) as env:
        assert env.action_spec.dtype == torch.float64
        assert env.state_spec["rng", "keys"].dtype == torch.int64
        assert env.state_spec["lifecycle", "reset_next_step"].dtype == torch.bool
    result = run_gate(os.environ.get('SE_REACHER_DEVICE', 'cpu'))
    assert result['execution_mode'] == 'DELEGATED_CPU'
    assert all(case.get('max_state_error', 0) == 0 for case in result['cases'])
