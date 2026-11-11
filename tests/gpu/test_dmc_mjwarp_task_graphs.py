"""Captured task formulas must read live physics and updated model metadata."""

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
        adapter.step(action)  # Warm and capture reward-only and full-output rounds.
        adapter.reset(91010)
        runtime, task = adapter._runtime, adapter._task
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
        adapter.step(action)
        checkpoint = adapter.get_state()
        runtime, task = adapter._runtime, adapter._task
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
