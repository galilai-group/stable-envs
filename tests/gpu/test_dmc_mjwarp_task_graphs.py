"""Captured task formulas must read live physics and updated model metadata."""

import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as se
from tests.gpu._shared import family_params, resolve_family
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
DMC_NAMES = [name for name, cls in se.registry.REGISTRY.items() if cls.family == "dmcontrol"]


def _unexpected_task_evaluation(*args, **kwargs):
    pytest.fail("A warmed task graph called a Python task method during replay")


def _eager_outputs(snapshot, methods, batch, device):
    values = {"observation": methods["observation"](snapshot), "reward": methods["reward"](snapshot)}
    info = methods["info"](snapshot)
    if info is not None:
        values["success"], values["info"] = info
    return TensorDict(values, [batch], device=device)


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_captured_task_outputs_match_eager_formulas_on_live_physics(name, monkeypatch):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp task graph")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        action = torch.full((3, adapter.action_dim), 0.125, device="cuda", dtype=torch.float64)
        runtime, task = adapter._runtime, adapter._task
        # Explicitly warm the single-round graphs: public steps may use a
        # separate paired graph instead.
        for reward_only in (True, False):
            runtime.advance(adapter._controls(action), task=task, reward_only=reward_only)
        adapter.reset(91010)
        methods = {name: getattr(task, name) for name in ("observation", "reward", "info")}
        for method in methods:
            monkeypatch.setattr(task, method, _unexpected_task_evaluation)

        for reward_only, value in ((True, -0.25), (False, 0.5), (True, 0.25), (False, -0.5)):
            controls = adapter._controls(action.fill_(value))
            actual = runtime.advance(controls, task=task, reward_only=reward_only)
            # Evaluate the original formulas on the very same post-step data.
            # This isolates task capture from long-horizon physics variability.
            snapshot = runtime.snapshot()
            if reward_only:
                assert_same_state(actual, methods["reward"](snapshot))
            else:
                assert_same_state(
                    TensorDict(actual, [3], device=adapter.device),
                    _eager_outputs(snapshot, methods, 3, adapter.device),
                )


def test_captured_task_reads_updated_native_metadata_without_recapture(monkeypatch):
    resolve_family(
        "DMCBallInCupCatch", family="dmc", modules=("dmcontrol._family",), subject="MJWarp task graph metadata"
    )
    with se.make("DMCBallInCupCatch", num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        action = torch.zeros((3, adapter.action_dim), device="cuda", dtype=torch.float64)
        checkpoint = adapter.get_state()
        runtime, task = adapter._runtime, adapter._task
        for reward_only in (True, False):
            runtime.advance(adapter._controls(action), task=task, reward_only=reward_only)
        if runtime.supports_pair:
            runtime.advance_pair(adapter._controls(action), task=task)
        storage = runtime._native_model_views["site_size"].data_ptr()
        for method in ("observation", "reward", "info"):
            monkeypatch.setattr(task, method, _unexpected_task_evaluation)

        # A zero target box can never contain the ball, while a 100-meter box
        # contains it in these states. Swap lane values to expose stale pointers
        # or accidental broadcasting after a model upload.
        for sizes in ((0.0, 100.0, 0.0), (100.0, 0.0, 100.0)):
            models = adapter._models.clone()
            models.columns["site_size"][:, task.target] = np.asarray(sizes)[:, None]
            runtime.load(checkpoint, models=models)
            assert runtime._native_model_views["site_size"].data_ptr() == storage
            expected = torch.tensor([float(size > 0) for size in sizes], device=adapter.device, dtype=torch.float64)
            for reward_only in (True, False):
                actual = runtime.advance(adapter._controls(action), task=task, reward_only=reward_only)
                if reward_only:
                    assert_same_state(actual, expected)
                else:
                    assert_same_state(actual["reward"], expected)
                    assert_same_state(actual["success"], expected)
            if runtime.supports_pair:
                first_reward, actual = runtime.advance_pair(adapter._controls(action), task=task)
                assert_same_state(first_reward, expected)
                assert_same_state(actual["reward"], expected)
                assert_same_state(actual["success"], expected)


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_reset_round_rewards_skip_discarded_outputs_and_match_full_reference(name, monkeypatch):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp reset-round outputs")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        actions = torch.full((3, adapter.action_dim), 0.125, device="cuda", dtype=torch.float64)
        control_step = adapter._control_step
        for all_reset in (False, True):
            initial = adapter.reset(91000)
            limits = [math.ceil(value) for value in adapter._limits]
            initial["lifecycle", "step_count"][:, 0] = torch.tensor(
                limits if all_reset else [limits[0] - 1, limits[1], 6], device="cuda"
            )
            initial["lifecycle", "reset_next_step"][:, 0] = torch.tensor(
                [True, True, True] if all_reset else [False, True, False], device="cuda"
            )
            initial["task", "cumulative_reward"][:, 0] = torch.tensor([11.0, 17.0, 23.0], device="cuda")
            initial["rng", "has_gauss"].fill_(True)
            initial["rng", "cached_gaussian"][:, 0] = torch.tensor([0.125, -0.375, 0.75], device="cuda")
            adapter.set_state(initial)

            def full_first_round(controls, *, reward_only=False, controls_validated=False):
                output = control_step(controls, controls_validated=controls_validated)
                return output["reward"] if reward_only else output

            with monkeypatch.context() as patch:
                patch.setattr(adapter, "_control_step", full_first_round)
                expected = adapter.step(actions)
            expected_native = adapter._source.get_state()
            expected_next_reset = adapter.reset(None)
            adapter.set_state(initial)
            phase = {"reward_only": False, "reward_rounds": 0}

            def checked_round(controls, *, reward_only=False, controls_validated=False):
                phase["reward_only"] = reward_only
                try:
                    output = control_step(controls, reward_only=reward_only, controls_validated=controls_validated)
                    if reward_only:
                        phase["reward_rounds"] += 1
                        assert output.shape == (3,) and output.dtype == torch.float64
                        reset_reward = output if all_reset else output[1:2]
                        assert_same_state(reset_reward, torch.zeros_like(reset_reward))
                    return output
                finally:
                    phase["reward_only"] = False

            def checked_formula(label, method):
                def evaluate(snapshot):
                    if phase["reward_only"]:
                        pytest.fail(f"Reset reward-only round evaluated discarded {label}")
                    return method(snapshot)

                return evaluate

            with monkeypatch.context() as patch:
                patch.setattr(adapter, "_control_step", checked_round)
                for method in ("observation", "info"):
                    patch.setattr(adapter._task, method, checked_formula(method, getattr(adapter._task, method)))
                actual = adapter.step(actions)
            assert phase["reward_rounds"] == 1
            assert_same_state(actual, expected)
            assert_same_state(adapter._source.get_state(), expected_native)
            assert_same_state(adapter.reset(None), expected_next_reset)


def test_failed_continuing_reward_round_preserves_pending_resets_and_recovers(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        initial = adapter.get_state()
        initial["lifecycle", "step_count"][:, 0] = torch.tensor(
            [math.ceil(adapter._limits[0]) - 1, math.ceil(adapter._limits[1]), 6], device="cuda"
        )
        initial["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False], device="cuda")
        adapter.set_state(initial)
        actions = torch.full((3, 1), 0.125, device="cuda", dtype=torch.float64)
        expected = adapter.step(actions)
        adapter.set_state(initial)
        native = adapter._source.get_state()
        physics = adapter._runtime.physical_state()
        counts, pending = adapter._counts.copy(), adapter._pending.copy()
        runtime = adapter._runtime
        check_result = runtime._check_result

        def fail_continuing_round(*, recorded=False):
            if recorded:
                runtime._data_view("qvel")[-1, -1] = float("nan")
                return check_result()
            return check_result(recorded=recorded)

        def unexpected_reset(*args, **kwargs):
            pytest.fail("Failed continuing physics consumed a pending lane's reset RNG")

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_check_result", fail_continuing_round)
            patch.setattr(adapter._source, "prepare_reset", unexpected_reset)
            with pytest.raises(RuntimeError, match="produced nonfinite qvel"):
                adapter.step(actions)
        assert_same_state(adapter.get_state(), initial)
        assert_same_state(adapter._source.get_state(), native)
        assert_same_state(adapter._runtime.physical_state(), physics)
        np.testing.assert_array_equal(adapter._counts, counts)
        np.testing.assert_array_equal(adapter._pending, pending)
        adapter.set_state(initial)
        assert_same_state(adapter.step(actions), expected)
    finally:
        adapter.close()
