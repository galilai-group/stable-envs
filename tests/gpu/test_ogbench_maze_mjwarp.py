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
