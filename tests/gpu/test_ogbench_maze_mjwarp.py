"""Maze lanes on MuJoCo Warp follow the bit-exact CPU batch to float32 rounding, with the same task bookkeeping."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
pytest.importorskip("mujoco_warp")

from stable_envs.envs.ogbench.simulation.maze_batch import OGBMazeBatchAdapter, OGBMazeWarpAdapter

pytestmark = pytest.mark.gpu

#: Largest |qpos| difference after one step from the same State (float32 physics against float64).
ONE_STEP_QPOS = 1e-4


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


@pytest.mark.parametrize("loco,maze", [("point", "medium"), ("ant", "medium"), ("humanoid", "medium")])
def test_reset_step_and_restore_follow_the_cpu_batch(loco, maze):
    lanes = 8
    cpu, gpu = (
        OGBMazeBatchAdapter(lanes, seed=4, loco=loco, maze=maze),
        OGBMazeWarpAdapter(lanes, "cuda", seed=4, loco=loco, maze=maze),
    )
    try:
        a, b = cpu.get_state(), gpu.get_state().cpu()
        for key in a.keys(True, True):
            if key[0] in ("task", "rng"):
                assert torch.equal(a[key], b[key]), key
            else:
                torch.testing.assert_close(b[key], a[key], rtol=0, atol=1e-5)
        actions = np.random.default_rng(0).uniform(-1, 1, (lanes, cpu.action_dim)).astype(np.float32)
        gpu.set_state(cpu.get_state().to("cuda"))
        want, got = cpu.step(actions), gpu.step(torch.as_tensor(actions, device="cuda"))
        assert got.device.type == "cuda"
        for key in ("observation", "reward", "terminated", "success"):
            assert got[key].dtype == want[key].dtype and got[key].shape == want[key].shape
        assert (got["info", "qpos"].cpu() - want["info", "qpos"]).abs().max() < ONE_STEP_QPOS
        saved = gpu.get_state()
        gpu.step(torch.as_tensor(actions, device="cuda"))
        gpu.set_state(saved)
        again = gpu.get_state()
        for key in saved.keys(True, True):
            assert torch.equal(saved[key], again[key]), key
    finally:
        cpu.close()
        gpu.close()


def test_teleports_draw_from_each_lanes_rng_as_the_source_does():
    lanes = 4
    cpu, gpu = (
        OGBMazeBatchAdapter(lanes, seed=3, maze="teleport"),
        OGBMazeWarpAdapter(lanes, "cuda", seed=3, maze="teleport"),
    )
    try:
        state = cpu.get_state()
        centre = torch.tensor(cpu.task.teleport["teleport_in_xys"][0], dtype=torch.float64)
        state["root", "position"] = centre + torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.1, 1.1], [-1.1, -1.1]], dtype=torch.float64
        )
        zero = np.zeros((lanes, 2), np.float32)
        cpu.set_state(state)
        gpu.set_state(state.to("cuda"))
        for _ in range(3):
            want, got = cpu.step(zero), gpu.step(torch.as_tensor(zero, device="cuda"))
            for key in want["state"].keys(True, True):
                if key[0] in ("task", "rng"):
                    assert torch.equal(got["state"][key].cpu(), want["state"][key]), key
            torch.testing.assert_close(
                got["state"]["root", "position"].cpu(), want["state"]["root", "position"], rtol=0, atol=1e-4
            )
    finally:
        cpu.close()
        gpu.close()


#: Largest |qpos| difference after 4 steps between a lane and its restored copy (contact order differs by run).
RESTORE_QPOS = 1e-3


@pytest.mark.parametrize("loco", ["point", "ant", "humanoid"])
def test_a_restored_state_resumes_to_rounding(loco):
    lanes = 16
    a, b = OGBMazeWarpAdapter(lanes, "cuda", seed=5, loco=loco), OGBMazeWarpAdapter(lanes, "cuda", seed=11, loco=loco)
    try:
        b.set_state(a.get_state())
        actions = torch.as_tensor(
            np.random.default_rng(6).uniform(-1, 1, (4, lanes, a.action_dim)), dtype=torch.float32, device="cuda"
        )
        for t in range(4):
            sa, sb = a.step(actions[t])["state"], b.step(actions[t])["state"]
        for key in sa.keys(True, True):
            if key[0] in ("task", "rng") or key == ("simulator", "time"):
                assert torch.equal(sa[key], sb[key]), key
            elif key[-1] == "position" or key[-1] == "orientation":
                assert (sa[key] - sb[key]).abs().max() <= RESTORE_QPOS, key
    finally:
        a.close()
        b.close()


def test_inactive_lanes_keep_their_state_and_last_outputs():
    lanes = 4
    gpu = OGBMazeWarpAdapter(lanes, "cuda", seed=2, loco="ant", maze="medium")
    try:
        actions = torch.as_tensor(
            np.random.default_rng(1).uniform(-1, 1, (2, lanes, gpu.action_dim)), dtype=torch.float32, device="cuda"
        )
        first = gpu.step(actions[0])
        mask = torch.tensor([True, False, True, False], device="cuda")
        second = gpu.step(actions[1], active=mask)
        for key in first["state"].keys(True, True):
            assert torch.equal(second["state"][key][~mask], first["state"][key][~mask]), key
        for key in ("observation", "reward", "success"):
            assert torch.equal(second[key][~mask], first[key][~mask]), key
        assert not torch.equal(second["observation"][mask], first["observation"][mask])
    finally:
        gpu.close()


def test_many_teleports_keep_the_rng_streams_of_the_cpu_batch():
    """More teleport draws than one draw-ahead table holds, with State reads in between."""
    lanes = 32
    cpu, gpu = (
        OGBMazeBatchAdapter(lanes, seed=3, maze="teleport"),
        OGBMazeWarpAdapter(lanes, "cuda", seed=3, maze="teleport"),
    )
    try:
        inbound = torch.tensor(cpu.task.teleport["teleport_in_xys"], dtype=torch.float64)
        rng = np.random.default_rng(0)
        for t in range(100):
            if t % 5 == 0:
                state = cpu.get_state()
                state["root", "position"] = inbound[torch.as_tensor(rng.integers(len(inbound), size=lanes))] + 0.2
                cpu.set_state(state)
                gpu.set_state(state.to("cuda"))
            actions = rng.uniform(-0.1, 0.1, (lanes, 2)).astype(np.float32)
            cpu.step(actions, capture_state=t % 3 == 0)
            gpu.step(torch.as_tensor(actions, device="cuda"), capture_state=t % 3 == 0)
        want, got = cpu.get_state(), gpu.get_state().cpu()
        for key in want.keys(True, True):
            if key[0] in ("task", "rng"):
                assert torch.equal(got[key], want[key]), key
    finally:
        cpu.close()
        gpu.close()
