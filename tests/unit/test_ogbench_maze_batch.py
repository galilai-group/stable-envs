"""Batched maze lanes on mujoco.batch give the source's State and outputs bit for bit."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
pytest.importorskip("mujoco.batch")

from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter
from stable_envs.envs.ogbench.simulation.maze_batch import OGBMazeBatchAdapter
from tests.helpers.reference.semantics import assert_same_state

BODIES = [
    ("point", "medium"),
    ("point", "giant"),
    ("point", "teleport"),
    ("ant", "medium"),
    ("ant", "giant"),
    ("ant", "teleport"),
    ("humanoid", "medium"),
    ("humanoid", "large"),
]


def _smooth(rng, lanes, steps, dim):
    """Slowly varying actions, so that bodies travel and reach teleports and walls."""
    phase = rng.uniform(0, 2 * np.pi, (lanes, 1, dim))
    t = np.arange(steps)[None, :, None]
    return np.clip(np.sin(0.05 * t + phase) + 0.3 * rng.standard_normal((lanes, steps, dim)), -1, 1)


@pytest.mark.parametrize("loco,maze", BODIES)
def test_batch_matches_source_lanes(loco, maze):
    lanes, steps = 6, 60
    batch, source = (
        OGBMazeBatchAdapter(lanes, seed=11, loco=loco, maze=maze),
        OGBMazeAdapter(lanes, seed=11, loco=loco, maze=maze),
    )
    try:
        assert_same_state(batch.get_state(), source.get_state())
        actions = _smooth(np.random.default_rng(4), lanes, steps, batch.action_dim).astype(np.float32)
        for t in range(steps):
            mask = None if t < 2 else torch.tensor([(t + lane) % 3 != 0 for lane in range(lanes)])
            assert_same_state(batch.step(actions[:, t], active=mask), source.step(actions[:, t], active=mask))
        assert_same_state(
            batch.step(actions[:, 0], capture_state=False), source.step(actions[:, 0], capture_state=False)
        )
        held = batch.get_state()
        assert_same_state(batch.reset(), source.reset())
        assert_same_state(batch.reset(), source.reset())
        for adapter in (batch, source):
            adapter.set_state(held)
        for t in range(5):
            assert_same_state(batch.step(actions[:, t]), source.step(actions[:, t]))
        assert_same_state(batch.reset(42), source.reset(42))
        assert_same_state(batch.step(actions[:, 1]), source.step(actions[:, 1]))
    finally:
        batch.close()
        source.close()


def test_teleport_lanes_teleport_as_the_source_does():
    lanes = 4
    batch, source = OGBMazeBatchAdapter(lanes, seed=3, maze="teleport"), OGBMazeAdapter(lanes, seed=3, maze="teleport")
    try:
        state = batch.get_state()
        centre = torch.tensor(batch.task.teleport["teleport_in_xys"][0], dtype=torch.float64)
        offsets = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.1, 1.1], [-1.1, -1.1]], dtype=torch.float64)
        state["root", "position"] = centre + offsets
        for adapter in (batch, source):
            adapter.set_state(state)
        zero = np.zeros((lanes, 2), np.float32)
        out = batch.step(zero)
        assert_same_state(out, source.step(zero))
        moved = (out["state"]["root", "position"] - state["root", "position"]).abs().sum(-1) > 0.5
        assert moved.tolist() == [True, True, False, False]
        drew = (out["state"]["rng", "global_position"] != state["rng", "global_position"]).reshape(-1)
        assert drew.tolist() == [True, True, False, False]
    finally:
        batch.close()
        source.close()


@pytest.mark.parametrize(
    "batch,device,backend,threshold,expected",
    [
        (1, "cpu", "default", None, "native"),
        (4096, "cpu", "default", None, "native"),
        (4096, "cpu", "auto", None, "native"),
        (1, "cuda", "default", None, "mjwarp"),
        (1, "cuda:1", "auto", 8, "native"),
        (7, "cuda", "auto", 8, "native"),
        (8, "cuda", "auto", 8, "mjwarp"),
        (4096, "cuda", "native", None, "native"),
        (1, "cuda", "mjwarp", None, "mjwarp"),
    ],
)
def test_maze_backend_routing(batch, device, backend, threshold, expected):
    from stable_envs.envs.ogbench.simulation.execution import maze_backend

    assert maze_backend(batch, device, "ant", backend, threshold) == expected


def test_auto_uses_the_measured_cutoff_of_each_body():
    from stable_envs.envs.ogbench.simulation.execution import MAZE_MJWARP_MIN_ENVS, maze_backend

    for loco, cutoff in MAZE_MJWARP_MIN_ENVS.items():
        assert maze_backend(cutoff - 1, "cuda", loco, "auto") == "native"
        assert maze_backend(cutoff, "cuda", loco, "auto") == "mjwarp"
    with pytest.raises(ValueError, match="requires device='cuda'"):
        maze_backend(8, "cpu", "point", "mjwarp")
