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


def test_legacy_streams_draw_as_random_state():
    from stable_envs.envs.ogbench.simulation.maze_batch import LegacyMT

    seeds = [0, 1, 17, 2**32 - 1, 123456789]
    mt = LegacyMT(len(seeds))
    rows = np.arange(len(seeds))
    mt.seed(rows, seeds)
    states = [np.random.RandomState(seed) for seed in seeds]
    # More than one block of 624 outputs, so every lane twists its key at least twice.
    for k in range(700):
        if k % 3 == 0:
            got, want = mt.randint(rows, 1, 6), [s.randint(1, 6) for s in states]
        elif k % 3 == 1:
            got, want = mt.uniform(rows, -1.0, 1.0), [s.uniform(low=-1.0, high=1.0) for s in states]
        else:
            got, want = mt.randint(rows[1:3], 0, 3), [s.randint(3) for s in states[1:3]]
        assert np.array_equal(got, np.asarray(want)), k
    for i, state in enumerate(states):
        _, keys, position, _, _ = state.get_state()
        assert np.array_equal(mt.key[i], keys) and mt.pos[i] == position


@pytest.mark.parametrize("maze", ["medium", "giant", "teleport"])
def test_the_wall_grid_gives_the_clearance_test_against_every_wall(maze):
    from stable_envs.envs.ogbench.simulation.maze import _make_maze, _rest_walls
    from stable_envs.envs.ogbench.simulation.maze_batch import RestGrid

    env = _make_maze("point", maze)
    try:
        lo, hi, radius, offset = rest = _rest_walls(env.model)
    finally:
        env.close()
    rng = np.random.default_rng(0)
    xy = rng.uniform(lo[:, :2].min(0) - 3, hi[:, :2].max(0) + 3, (20000, 2))
    # Points on the clearance boundary of wall faces and corners.
    edge = np.concatenate([lo[:, :2] - radius - 1e-3, hi[:, :2] + radius + 1e-3])
    xy = np.concatenate([xy, edge, edge + 1e-12, edge - 1e-12])
    centre = np.zeros((len(xy), 3))
    centre[:, :2] = xy
    centre += offset
    near = np.clip(centre[:, None], lo, hi) - centre[:, None]
    want = (near * near).sum(-1).min(1) > (radius + 1e-3) ** 2
    got = RestGrid(*rest).resting(xy)
    assert np.array_equal(got, want) and want.any() and not want.all()


def test_cuda_outputs_of_the_cpu_batch_equal_its_cpu_outputs():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cpu, cuda = OGBMazeBatchAdapter(4, "cpu", seed=3, loco="ant"), OGBMazeBatchAdapter(4, "cuda", seed=3, loco="ant")
    try:
        actions = np.random.default_rng(0).uniform(-1, 1, (4, 8)).astype(np.float32)
        want, got = cpu.step(actions), cuda.step(actions)
        assert got.device.type == "cuda"
        assert_same_state(got.cpu(), want)
        assert_same_state(cuda.get_state().cpu(), cpu.get_state())
    finally:
        cpu.close()
        cuda.close()
