"""Core contracts for explicit CUDA MJWarp stepping across registered DMC tasks.

These tests cover integration and checkpoint semantics, not calibrated long-horizon
agreement with native MuJoCo. Run scripts/ci/run_dmc_mjwarp.py to require CUDA and
reject skipped cases.
"""

import math

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.gpu._shared import family_params, resolve_family
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
DMC_NAMES = [name for name, cls in se.registry.REGISTRY.items() if cls.family == "dmcontrol"]


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_cuda_stepping_snapshot_ownership_and_pixels(name):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp stepping")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        assert env.simulation_backend == env._simulator.simulation_backend == "mjwarp"
        assert env.execution_mode == "MUJOCO_WARP"
        env.reset(seed=31)
        actions = torch.zeros(3, 2 * env.action_dim, device="cuda", dtype=torch.float64)[:, ::2]
        assert not actions.is_contiguous()
        env.step(actions, states=None, frames=None)
        saved = env.get_state()
        saved["rng", "has_gauss"][:] = True
        saved["rng", "cached_gaussian"][:, 0] = torch.tensor(
            [0.123456789012345, -0.75, 1.25], device="cuda", dtype=torch.float64
        )
        if name == "DMCQuadrupedWalk":
            assert saved["activation"].shape[1] > 0
            saved["activation"][:] = torch.tensor([0.15, 0.45, 0.85], device="cuda", dtype=torch.float64)[:, None]
        env.set_state(saved)
        assert_same_state(env.get_state(), saved)
        pixels = env.render().clone()
        result = env.step(actions, states=None, frames=None)
        held = result.clone()
        env.step(actions.fill_(0.25), states=None, frames=None)
        assert all(leaf.device.type == "cuda" for leaf in result.values(True, True))
        assert_same_state(result, held)
        env.set_state(saved)
        assert_same_state(env.get_state(), saved)
        assert_same_state(env.render(), pixels)
        assert_same_state(env.get_state(), saved)

        supplied = saved.clone()
        env.set_state(supplied)
        supplied["qpos"].add_(1)
        exported = env.get_state()
        exported["qvel"].add_(1)
        assert_same_state(env.get_state(), saved)
        sequence = torch.zeros(3, 2, env.action_dim, device="cuda", dtype=torch.float64)
        captured = env.step(sequence, states=slice(None), frames=slice(None))
        before_render = env.get_state()
        for step in range(2):
            assert_same_state(env.render(captured["states"][:, step]), captured["frames"][:, step])
        assert_same_state(env.get_state(), before_render)


def test_mixed_resets_in_both_action_repeats_match_native_lifecycle():
    from stable_envs.envs.dmcontrol.simulation.factory import make_local
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    source = make_local("cartpole-balance", 3, "cpu", 17)
    try:
        state = source.get_state()
        limit = math.ceil(source.envs[0]._env._step_limit)
        # Lane 0 resets in the second control round, lane 1 in the first;
        # lane 2 continues through both rounds.
        state["lifecycle", "step_count"][:] = torch.tensor([[limit - 1], [limit], [6]])
        state["lifecycle", "reset_next_step"][:] = torch.tensor([[False], [True], [False]])
        state["task", "cumulative_reward"][:] = torch.tensor([[11.0], [17.0], [23.0]])
        state["rng", "has_gauss"][:2] = True
        state["rng", "cached_gaussian"][:2, 0] = torch.tensor([0.125, -0.375], dtype=torch.float64)
        adapter.set_state(state.to("cuda"))
        source.set_state(state)
        actions = torch.full((3, 1), 0.25, dtype=torch.float64)
        expected = source.step(actions)
        actual = adapter.step(actions.to("cuda")).cpu()
        assert_same_state(actual["state", "rng"], expected["state", "rng"])
        assert_same_state(actual["state", "lifecycle"], expected["state", "lifecycle"])
        assert actual["state", "lifecycle", "step_count"].flatten().tolist() == [0, 1, 8]
        # Cartpole's first reset velocity consumes the cached Gaussian. Lane 0
        # has not integrated after its reset, so this value must remain exact.
        assert actual["state", "qvel"][0, 0].item() == 0.01 * 0.125
        for key in ("qpos", "qvel"):
            torch.testing.assert_close(actual["state", key], expected["state", key], atol=1e-5, rtol=0)
        torch.testing.assert_close(actual["observation"], expected["observation"], atol=1e-5, rtol=0)
        torch.testing.assert_close(actual["reward"], expected["reward"], atol=1e-6, rtol=0)
        torch.testing.assert_close(
            actual["state", "task", "cumulative_reward"],
            state["task", "cumulative_reward"] + actual["reward"][:, None],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(actual["score"], expected["score"], atol=1e-8, rtol=0)
        # An ordinary step after reconstruction advances the persistent batch,
        # without redrawing RNG or resetting the cumulative wrapper reward.
        following = adapter.step(actions.to("cuda"), capture_state=False)
        assert "state" not in following.keys()
        after = adapter.get_state().cpu()
        assert after["lifecycle", "step_count"].flatten().tolist() == [2, 3, 10]
        assert_same_state(after["rng"], actual["state", "rng"])
        torch.testing.assert_close(
            after["task", "cumulative_reward"],
            actual["state", "task", "cumulative_reward"] + following["reward"].cpu()[:, None],
            atol=0,
            rtol=0,
        )
    finally:
        adapter.close()
        source.close()


def test_invalid_later_lane_restore_preserves_models_state_and_rng():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 5)
    try:
        before = adapter.get_state()
        native_before = adapter._source.get_state()
        masses = [model.body_mass.copy() for model in adapter._models]
        bad = before.clone()
        bad["physics", "agent", "cart_mass"][0] = 1.25
        bad["physics", "agent", "cart_mass"][2] = -1
        with pytest.raises(ValueError):
            adapter.set_state(bad)
        assert_same_state(adapter.get_state(), before)
        assert_same_state(adapter._source.get_state(), native_before)
        for model, mass in zip(adapter._models, masses, strict=True):
            np.testing.assert_array_equal(model.body_mass, mass)
        reset_after_failure = adapter.reset(None)
        adapter.set_state(before)
        assert_same_state(adapter.reset(None), reset_after_failure)
    finally:
        adapter.close()
