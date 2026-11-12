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


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_mutating_returned_leaves_preserves_live_state_and_physics(name):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp output ownership")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        actions = torch.full((3, adapter.action_dim), 0.125, device="cuda", dtype=torch.float64)
        for mixed in (False, True):
            state = adapter.reset(91000)
            if mixed:
                limit = math.ceil(adapter._limits[0])
                state["lifecycle", "step_count"][:, 0] = torch.tensor([limit - 1, limit, 6], device="cuda")
                state["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False], device="cuda")
                adapter.set_state(state)
            output = adapter.step(actions, capture_state=mixed)
            checkpoint = adapter.get_state()
            physics = adapter._runtime.physical_state()
            if mixed and "info" in output.keys():
                saved_qpos = output["state", "qpos"].clone()
                output["info", "qpos"].fill_(123)
                assert_same_state(output["state", "qpos"], saved_qpos)
            for value in output.values(True, True):
                if value.dtype == torch.bool:
                    value.logical_not_()
                else:
                    value.fill_(123)
            assert_same_state(adapter.get_state(), checkpoint)
            assert_same_state(adapter._runtime.physical_state(), physics)
            held = output.clone()
            adapter.step(actions, capture_state=False)
            adapter.reset(91020)
            assert_same_state(output, held)


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


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_reward_only_first_round_matches_full_outputs(name, monkeypatch):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp reward-only round")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        checkpoint = env.reset(seed=91000)
        actions = np.random.default_rng(91000 + 91717).uniform(-1, 1, (5, 3, env.action_dim))
        actions = torch.as_tensor(actions, device="cuda", dtype=torch.float64)
        # Warm the public path before comparing its replay behavior.
        adapter.step(actions[0])
        adapter.set_state(checkpoint)
        control_step = adapter._control_step

        # Reference path materializes all outputs in both rounds, as before the
        # optimization. Retain every leaf to check reward, state and ownership.
        def full_outputs(controls, *, reward_only=False, controls_validated=False):
            output = control_step(controls, controls_validated=controls_validated)
            return output["reward"] if reward_only else output

        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "supports_pair", False)
            patch.setattr(adapter, "_control_step", full_outputs)
            expected = [adapter.step(action) for action in actions]

        adapter.set_state(checkpoint)
        # The forced checked reference releases the inactive paired graph.
        # Warm the selected public route again before measuring Python calls.
        adapter.step(actions[0])
        adapter.set_state(checkpoint)
        calls = {"observation": 0, "reward": 0, "info": 0}

        def count_calls(name, method):
            def evaluate(snapshot):
                calls[name] += 1
                return method(snapshot)

            return evaluate

        with monkeypatch.context() as patch:
            for method in calls:
                patch.setattr(adapter._task, method, count_calls(method, getattr(adapter._task, method)))
            actual = [adapter.step(action) for action in actions]
        assert calls == {"observation": 0, "reward": 0, "info": 0}
        for got, want in zip(actual, expected, strict=True):
            assert_same_state(got, want)
        # Mutating a returned output must not change the live checkpoint.
        live = adapter.get_state()
        actual[-1]["state", "qpos"].add_(1)
        actual[-1]["reward"].zero_()
        assert_same_state(adapter.get_state(), live)


def test_failed_second_round_keeps_first_round_checkpoint(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        # This checks the per-round fallback. Paired device failures and the
        # conditional second-round guard have their own tests.
        monkeypatch.setattr(adapter._runtime, "supports_pair", False)
        initial = adapter.get_state()
        actions = torch.full((3, 1), 0.25, dtype=torch.float64, device="cuda")
        adapter._control_step(adapter._controls(actions))
        after_first = adapter.get_state()
        adapter.set_state(initial)
        advance = adapter._runtime.advance
        rounds = 0

        def fail_second(controls, **kwargs):
            nonlocal rounds
            rounds += 1
            result = advance(controls, **kwargs)
            if rounds == 2:
                raise RuntimeError("injected second-round failure")
            return result

        monkeypatch.setattr(adapter._runtime, "advance", fail_second)
        with pytest.raises(RuntimeError, match="injected second-round failure"):
            adapter.step(actions)
        assert rounds == 2
        assert_same_state(adapter.get_state(), after_first)
    finally:
        adapter.close()


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


def test_invalid_actions_and_scaled_controls_preserve_pending_resets(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        initial = adapter.get_state()
        initial["lifecycle", "reset_next_step"].fill_(True)
        adapter.set_state(initial)
        native_before = adapter._source.get_state()
        physics_before = adapter._runtime.physical_state()
        controls = torch.zeros(3, 1, dtype=torch.float32, device="cuda")
        controls[2] = float("inf")
        # Direct runtime callers retain validation by default.
        with pytest.raises(ValueError, match="Controls must be finite"):
            adapter._runtime.advance(controls)
        with monkeypatch.context() as patch:

            def unexpected_advance(*args, **kwargs):
                pytest.fail("Invalid controls reached physics")

            patch.setattr(adapter._runtime, "advance", unexpected_advance)
            for bad in (float("nan"), float("inf"), 1e300):
                actions = torch.zeros(3, 1, dtype=torch.float64, device="cuda")
                actions[2] = bad
                with pytest.raises(ValueError, match="finite"):
                    adapter.step(actions)
                assert_same_state(adapter.get_state(), initial)
                assert_same_state(adapter._source.get_state(), native_before)
                assert_same_state(adapter._runtime.physical_state(), physics_before)
        # Validation errors must leave the CUDA context and reset path usable.
        adapter.step(torch.zeros(3, 1, device="cuda"))
        assert adapter.get_state()["lifecycle", "step_count"].flatten().tolist() == [1, 1, 1]
    finally:
        adapter.close()


@pytest.mark.parametrize("failed_round", [1, 2])
def test_nonfinite_round_preserves_last_valid_checkpoint(failed_round, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        monkeypatch.setattr(adapter._runtime, "supports_pair", False)
        initial = adapter.get_state()
        actions = torch.full((3, 1), 0.25, dtype=torch.float64, device="cuda")
        adapter._control_step(adapter._controls(actions))
        after_first = adapter.get_state()
        adapter.set_state(initial)
        expected = adapter.step(actions)
        adapter.set_state(initial)
        check_result = adapter._runtime._check_result
        rounds = 0

        def fail_result(*, recorded=False):
            nonlocal rounds
            if recorded:
                rounds += 1
                if rounds == failed_round:
                    adapter._runtime._data_view("qvel")[-1, -1] = float("nan")
                    return check_result()
            return check_result(recorded=recorded)

        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "_check_result", fail_result)
            with pytest.raises(RuntimeError, match="produced nonfinite qvel"):
                adapter.step(actions)
        assert rounds == failed_round
        assert_same_state(adapter.get_state(), initial if failed_round == 1 else after_first)
        adapter.set_state(initial)
        # A stale error bit must be cleared by the captured graph on replay.
        with adapter._runtime._scope():
            adapter._runtime._result_error.fill_(127)
        assert_same_state(adapter.step(actions), expected)
    finally:
        adapter.close()


def test_result_flags_diagnose_all_fields_and_capacity_on_cuda():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
    from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import _RESULT_FIELDS

    adapter = DMCWarpAdapter("quadruped", 3, "cuda", 17)
    try:
        from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import mjw

        runtime = adapter._runtime
        with runtime._scope():
            for name in _RESULT_FIELDS:
                view = runtime._data_view(name).reshape(-1)
                original = view[-1].clone()
                view[-1] = float("inf")
                with pytest.raises(RuntimeError, match=f"produced nonfinite {name}"):
                    runtime._check_result()
                view[-1] = original
                runtime._check_result()
            overflow = runtime._data_view("overflow")
            original = overflow.clone()
            for flag in mjw.OverflowType:
                overflow.zero_()
                overflow[-1] = int(flag)
                if int(flag) & runtime._capacity_overflow:
                    with pytest.raises(RuntimeError, match="capacity overflow"):
                        runtime._check_result()
                else:
                    runtime._check_result()
            overflow.copy_(original)
            nacon = runtime._data_view("nacon")
            original_count = nacon.clone()
            nacon.fill_(runtime._d.naconmax + 1)
            with pytest.raises(RuntimeError, match="capacity overflow"):
                runtime._check_result()
            nacon.copy_(original_count)
            runtime._check_result()
    finally:
        adapter.close()


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_pooled_resets_preserve_rng_targets_and_variation(name):
    from stable_envs.envs.dmcontrol.simulation.factory import make_local
    from stable_envs.envs.dmcontrol.simulation.physics import LEAVES
    from stable_envs.envs.dmcontrol.simulation.state import MODEL_FIELDS

    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp pooled resets")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        source = make_local(adapter.key, 3, "cpu", 91000)
        try:
            assert len(adapter._source.envs) == 1
            leaves = LEAVES.get(adapter.key, ())
            if leaves:
                values = {leaf.path: [[leaf.low], [leaf.default], [leaf.high]] for leaf in leaves}
                adapter.configure_physics(values)
                source.configure_physics(values)
            state = source.reset(91020)
            assert_same_state(adapter.reset(91020).cpu(), state)
            limit = math.ceil(source.envs[0]._env._step_limit)
            state["lifecycle", "step_count"][:, 0] = torch.tensor([limit - 1, limit, 6])
            state["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False])
            state["rng", "has_gauss"].fill_(True)
            state["rng", "cached_gaussian"][:, 0] = torch.tensor([0.125, -0.375, 0.75], dtype=torch.float64)
            source.set_state(state)
            adapter.set_state(state.to("cuda"))
            actions = torch.zeros(3, adapter.action_dim, dtype=torch.float64)
            for _ in range(3):
                expected = source.step(actions)["state"]
                actual = adapter.step(actions.to("cuda"))["state"].cpu()
                for section in ("rng", "lifecycle"):
                    assert_same_state(actual[section], expected[section])
                for key in source.fields:
                    if key in MODEL_FIELDS or (isinstance(key, tuple) and key[0] == "physics"):
                        assert_same_state(actual[key], expected[key])
            # Public reset(None) must continue each stream after mixed resets.
            assert_same_state(adapter.reset(None).cpu(), source.reset(None))
        finally:
            source.close()
