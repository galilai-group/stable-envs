"""Cube/Scene on MuJoCo Warp against OGBench: the scripted Cube oracle and a bound on Scene drift."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu]

pytest.importorskip("ogbench")
pytest.importorskip("mujoco_warp")

#: Smooth-action drift bounds from the same reset (float32 physics). On an L40S the largest gaps over 8
#: environments were 6.3e-4 rad and 2.6e-4 m in the first 25 steps, and 0.087 rad and 0.045 m in 200 steps,
#: where contacts with the drawer and the buttons make the paths separate; MuJoCo Warp is not deterministic
#: from run to run, so the 200-step bounds leave room.
SCENE_EARLY_STEPS, SCENE_EARLY_ARM_BOUND, SCENE_EARLY_EFFECTOR_BOUND = 25, 5e-3, 2e-3
SCENE_ARM_QPOS_BOUND = 0.5
SCENE_EFFECTOR_BOUND = 0.2


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("MuJoCo Warp Cube/Scene requires CUDA")


def _oracle_info(env):
    from ogbench.manipspace import lie

    info = env.compute_ob_info()
    target = env._cube_target_mocap_ids[0]
    info["privileged/target_block"] = 0
    info["privileged/target_block_pos"] = env._data.mocap_pos[target].copy()
    info["privileged/target_block_yaw"] = np.array([lie.SO3(wxyz=env._data.mocap_quat[target]).compute_yaw_radians()])
    return info


def _first(success):
    return [int(np.argmax(column)) if column.any() else -1 for column in success.T]


def test_cube_oracle_replay_has_ogbench_successes():
    _require_cuda()
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
    from stable_envs.envs.ogbench.simulation.manipulation_warp import OGBManipulationWarpAdapter

    lanes, steps = 8, 200
    ref = OGBManipulationAdapter("Cube", lanes, seed=7)
    gpu = OGBManipulationWarpAdapter("Cube", lanes, "cuda", seed=7)
    try:
        for episode in range(2):
            seed = 1000 + 37 * episode
            ref.reset(seed)
            gpu.reset(seed)
            oracles = []
            for lane, env in enumerate(ref.envs):
                np.random.seed(seed + lane)
                oracle = CubeMarkovOracle(env=env)
                oracle.reset(None, _oracle_info(env))
                oracles.append(oracle)
            want = np.zeros((steps, lanes), bool)
            got = np.zeros((steps, lanes), bool)
            for t in range(steps):
                action = np.stack([o.select_action(None, _oracle_info(env)) for o, env in zip(oracles, ref.envs)])
                action = torch.as_tensor(action, dtype=torch.float32)
                want[t] = ref.step(action, capture_state=False)["success"].numpy()
                got[t] = gpu.step(action.cuda(), capture_state=False)["success"].cpu().numpy()
            assert want.any(0).tolist() == got.any(0).tolist()
            assert _first(want) == _first(got)
    finally:
        ref.close()
        gpu.close()


def test_scene_smooth_actions_stay_near_ogbench():
    _require_cuda()
    from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
    from stable_envs.envs.ogbench.simulation.manipulation_warp import OGBManipulationWarpAdapter

    lanes, steps = 8, 200
    ref = OGBManipulationAdapter("Scene", lanes, seed=3)
    gpu = OGBManipulationWarpAdapter("Scene", lanes, "cuda", seed=3)
    env = ref.envs[0]
    arm = np.asarray(env.model.jnt_qposadr[env._arm_joint_ids])
    try:
        arm_gap = effector_gap = 0.0
        for t in range(steps):
            if t == SCENE_EARLY_STEPS:
                assert arm_gap <= SCENE_EARLY_ARM_BOUND
                assert effector_gap <= SCENE_EARLY_EFFECTOR_BOUND
            phase = t / 25 + np.arange(5)[None] + 0.7 * np.arange(lanes)[:, None]
            action = torch.as_tensor(0.6 * np.sin(phase), dtype=torch.float32)
            want = ref.step(action)["state"]
            got = gpu.step(action.cuda())["state"].cpu()
            arm_gap = max(
                arm_gap, float((got["physics", "qpos"][:, arm] - want["physics", "qpos"][:, arm]).abs().max())
            )
            effector = got["controller", "site_xpos"][:, 0] - want["controller", "site_xpos"][:, 0]
            effector_gap = max(effector_gap, float(effector.norm(dim=-1).max()))
        assert arm_gap <= SCENE_ARM_QPOS_BOUND
        assert effector_gap <= SCENE_EFFECTOR_BOUND
    finally:
        ref.close()
        gpu.close()


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_warp_state_has_the_cpu_leaves(family):
    _require_cuda()
    from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
    from stable_envs.envs.ogbench.simulation.manipulation_warp import OGBManipulationWarpAdapter

    ref = OGBManipulationAdapter(family, 3, seed=1)
    gpu = OGBManipulationWarpAdapter(family, 3, "cuda", seed=1)
    try:
        want, got = ref.get_state(), gpu.get_state()
        assert set(got.keys(True, True)) == set(want.keys(True, True))
        for key in want.keys(True, True):
            assert got[key].shape == want[key].shape and got[key].dtype == want[key].dtype, key
            assert got[key].device.type == "cuda"
        gpu.step(torch.zeros(3, 5, device="cuda"))
        saved = gpu.get_state()
        after = gpu.step(torch.full((3, 5), 0.3, device="cuda"))["state"]
        gpu.set_state(saved)
        again = gpu.step(torch.full((3, 5), 0.3, device="cuda"))["state"]
        qpos = (after["physics", "qpos"] - again["physics", "qpos"]).abs().max()
        assert float(qpos) < 1e-4
    finally:
        ref.close()
        gpu.close()
