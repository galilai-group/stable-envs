"""Bounded full-State recovery gate for the exact historical OGBMaze default."""

import os

import pytest
import torch

import stable_envs as se

pytest.importorskip("ogbench")
pytest.importorskip("stable_worldmodel")

from tests.helpers.reference.gates.ogbench_maze import run_gate


def test_ogbench_maze_full_state_semantics():
    with se.make("PointMazeLargeNavigate", device="cpu", num_envs=1) as env:
        assert env.action_spec.dtype == torch.float32
        assert env.state_spec["root", "position"].dtype == torch.float64
        assert env.state_spec["rng", "env_state"].dtype == torch.uint8
        assert env.state_spec["rng", "global_has_gauss"].dtype == torch.bool
    result = run_gate(os.environ.get("SE_OGBENCH_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])
