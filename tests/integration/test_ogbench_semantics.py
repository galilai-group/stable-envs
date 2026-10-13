"""Maze full-State, per-cell and appearance gates against the official env and stable-worldmodel."""

import os

import pytest
import torch

import stable_envs as se

pytest.importorskip("ogbench")
from stable_envs.envs.ogbench.maze import CELL_CLASSES  # noqa: E402
from tests.helpers.reference.gates.ogbench_maze import (  # noqa: E402
    FRAME_MSE_MAX,
    PIXEL_CASES,
    STATE_CASES,
    appearance_gate,
    cell_gate,
    pixel_appearance_gate,
    run_gate,
)


@pytest.mark.swm
def test_ogbench_maze_full_state_semantics():
    with se.make("PointMazeLargeNavigate", device="cpu", num_envs=1) as env:
        assert env.action_spec.dtype == torch.float32
        assert env.state_spec["root", "position"].dtype == torch.float64
        assert env.state_spec["rng", "env_state"].dtype == torch.uint8
        assert env.state_spec["rng", "global_has_gauss"].dtype == torch.bool
    result = run_gate(os.environ.get("SE_OGBENCH_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])


@pytest.mark.parametrize("cell", sorted(CELL_CLASSES))
def test_every_maze_cell_matches_the_official_env(cell):
    evidence = cell_gate(cell)
    assert len(evidence["cases"]) >= 2


@pytest.mark.swm
@pytest.mark.parametrize("case", sorted(STATE_CASES))
def test_ogbmaze_appearance_matches_swm_in_the_state_scene(case):
    assert appearance_gate(case)["state_exact"]


@pytest.mark.swm
@pytest.mark.parametrize("case", sorted(PIXEL_CASES))
@pytest.mark.parametrize("cell", ["PointMazeMediumNavigate", "AntMazeLargeNavigate", "HumanoidMazeMediumNavigate"])
def test_ogbmaze_appearance_matches_swm_in_the_pixel_scene(cell, case):
    assert pixel_appearance_gate(cell, case)["first_frame_mse"]["worst"] <= FRAME_MSE_MAX
