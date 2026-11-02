"""DMC Warp lifecycle tested independently of GPU arithmetic with native physics."""

import importlib
import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_local


class NativeSnapshot:
    """Same view contract as WarpSnapshot, backed by isolated native MjData."""

    def __init__(self, runtime):
        self.runtime = runtime

    def data(self, name):
        return torch.from_numpy(np.stack([getattr(data, name).copy() for data in self.runtime.datas]))

    def model(self, name):
        return torch.from_numpy(np.stack([getattr(model, name).copy() for model in self.runtime.models]))


class NativeRuntime:
    """Exact CPU engine fake: adapter resets/RNG/state logic remains under test."""

    def __init__(self, models, device, nsubsteps):
        assert torch.device(device).type == "cpu"
        self.models, self.nsubsteps = list(models), nsubsteps
        self.num_envs = len(self.models)
        self.datas = [mujoco.MjData(model) for model in self.models]
        self.fields = {"qpos": "qpos", "qvel": "qvel", "warmstart": "qacc_warmstart", "time": "time"}
        if self.models[0].na:
            self.fields["activation"] = "act"
        self.closed = False

    def validate_models(self, models):
        assert len(models) == self.num_envs
        for model in models:
            for owner in (model, model.opt, model.stat):
                for name in dir(owner):
                    if name.startswith("_"):
                        continue
                    value = getattr(owner, name)
                    if isinstance(value, (float, np.floating)) or (
                        isinstance(value, np.ndarray) and value.dtype.kind == "f"
                    ):
                        with np.errstate(over="ignore", invalid="ignore"):
                            if not np.isfinite(np.asarray(value, dtype=np.float32)).all():
                                raise ValueError(f"Model field {name} is not representable in float32")

    def load(self, state, models=None):
        assert not self.closed
        if models is not None:
            self.validate_models(models)
            self.models = list(models)
        for i, (model, data) in enumerate(zip(self.models, self.datas, strict=True)):
            mujoco.mj_resetData(model, data)
            for key, name in self.fields.items():
                if name == "time":
                    data.time = state[key][i].item()
                else:
                    getattr(data, name)[:] = state[key][i].numpy()
            # dm_control.Physics.forward disables actuation when reconstructing
            # a State. This fake tests lifecycle against that exact CPU oracle.
            disabled = model.opt.disableflags
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
            try:
                mujoco.mj_forward(model, data)
            finally:
                model.opt.disableflags = disabled
            data.qacc_warmstart[:] = state["warmstart"][i].numpy()

    def advance(self, controls):
        assert not self.closed
        for i, (model, data) in enumerate(zip(self.models, self.datas, strict=True)):
            data.ctrl[:] = controls[i].numpy()
            if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
                mujoco.mj_step(model, data, self.nsubsteps)
            else:
                mujoco.mj_step2(model, data)
                if self.nsubsteps > 1:
                    mujoco.mj_step(model, data, self.nsubsteps - 1)
            mujoco.mj_step1(model, data)
        return self.snapshot()

    def physical_state(self):
        assert not self.closed
        return TensorDict(
            {
                key: torch.from_numpy(np.stack([np.atleast_1d(getattr(data, name)).copy() for data in self.datas]))
                for key, name in self.fields.items()
            },
            [self.num_envs],
        )

    def snapshot(self):
        assert not self.closed
        return NativeSnapshot(self)

    def close(self):
        self.closed = True


@pytest.fixture
def adapter_type(monkeypatch):
    module = importlib.import_module("stable_envs.envs.dmcontrol.simulation.mjwarp")
    monkeypatch.setattr(module, "_require_cuda", lambda device: None)
    monkeypatch.setattr(module, "WarpRuntime", NativeRuntime)
    return module.DMCWarpAdapter


def assert_tree(actual, expected, *, exact=False):
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key, value in actual.items(True, True):
        tolerance = 0 if exact else 1e-6 if value.dtype == torch.float32 else 2e-11
        torch.testing.assert_close(value, expected[key], atol=tolerance, rtol=tolerance, equal_nan=True, msg=str(key))


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_staggered_resets_and_rewards_match_the_source(adapter_type, key):
    adapter, source = adapter_type(key, 3, "cpu", 17), make_local(key, 3, "cpu", 17)
    try:
        state = adapter.get_state()
        limit = math.ceil(source.envs[0]._env._step_limit)
        # Lane 0 resets on the second repeat; lane 1 resets on the first repeat.
        state["lifecycle", "step_count"][:] = torch.tensor([[limit - 1], [limit], [6]])
        state["lifecycle", "reset_next_step"][:] = torch.tensor([[False], [True], [False]])
        state["task", "cumulative_reward"][:] = torch.tensor([[11.0], [17.0], [23.0]])
        state["rng", "has_gauss"][1] = True
        state["rng", "cached_gaussian"][1] = 0.125
        adapter.set_state(state)
        source.set_state(state)
        assert_tree(adapter.get_state(), state, exact=True)
        actions = torch.full((3, adapter.action_dim), 0.25, dtype=torch.float64)
        if key == "quadruped":
            # Its asymmetric actuator ranges map 0.25 to nonrepresentable
            # float32 controls (e.g. 0.2). Isolate lifecycle from that deliberate
            # precision change by choosing exactly zero physical control.
            spec = source.envs[0]._env.action_spec()
            low = torch.from_numpy(spec.minimum.copy())
            scale = torch.from_numpy((spec.maximum - spec.minimum).copy()) / 2
            actions[:] = -low / scale - 1
        actual, expected = adapter.step(actions), source.step(actions)
        assert_tree(actual, expected)
        assert actual["state", "lifecycle", "step_count"].flatten().tolist() == [0, 1, 8]
        assert not actual["state", "lifecycle", "reset_next_step"].any()
        assert_tree(adapter.step(actions), source.step(actions))
        # Restoring the cached Gaussian and all episode fields repeats the same reset draws.
        adapter.set_state(state)
        assert_tree(adapter.step(actions), actual, exact=True)
    finally:
        adapter.close()
        source.close()


def test_capture_state_false_owned_snapshots_and_strided_actions(adapter_type):
    adapter = adapter_type("reacher", 3, "cpu", 11)
    try:
        state = adapter.get_state()
        retained = state.clone()
        actions = torch.full((3, 4), 0.25, dtype=torch.float64)[:, ::2]
        assert not actions.is_contiguous()
        out = adapter.step(actions, capture_state=False)
        assert "state" not in out.keys()
        after = adapter.get_state()
        adapter.set_state(state)
        full = adapter.step(actions.contiguous())
        assert_tree(out, full.exclude("state"), exact=True)
        assert_tree(after, full["state"], exact=True)
        frozen = full.clone()
        adapter.step(actions)
        adapter.reset(47)
        assert_tree(state, retained, exact=True)
        assert_tree(full, frozen, exact=True)
        live = adapter.get_state()
        baseline = live.clone()
        for value in live.values(True, True):
            value.zero_()
        assert_tree(adapter.get_state(), baseline, exact=True)
    finally:
        adapter.close()
    adapter.close()
    for operation in (adapter.get_state, adapter.reset, lambda: adapter.step(actions, capture_state=False)):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize(
    "fault", ["missing", "nonfinite", "float32_overflow", "dtype", "rng_position", "rng_key", "negative_count"]
)
def test_invalid_restore_does_not_change_any_lane(adapter_type, fault):
    adapter = adapter_type("finger", 3, "cpu", 5)
    try:
        before = adapter.get_state()
        bad = before.clone()
        if fault == "missing":
            bad = bad.exclude("qvel")
        elif fault == "nonfinite":
            bad["qpos"][2, 0] = float("nan")
        elif fault == "float32_overflow":
            bad["qvel"][2, 0] = 1e300
        elif fault == "dtype":
            bad["qvel"] = bad["qvel"].float()
        elif fault == "rng_position":
            bad["rng", "position"][2] = 625
        elif fault == "rng_key":
            bad["rng", "keys"][2, 0] = -1
        else:
            bad["lifecycle", "step_count"][2] = -1
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(bad)
        assert_tree(adapter.get_state(), before, exact=True)
        # Failure must also leave hidden native RNG/model state unchanged.
        adapter.reset(None)
        reset_after_failure = adapter.get_state()
        adapter.set_state(before)
        assert_tree(adapter.reset(None), reset_after_failure, exact=True)
    finally:
        adapter.close()


@pytest.mark.parametrize("fault", ["episode_target_overflow", "negative_physics", "derived_physics_overflow"])
def test_invalid_model_restore_preserves_native_models_and_live_state(adapter_type, fault):
    adapter = adapter_type("reacher", 3, "cpu", 5)
    try:
        before = adapter.get_state()
        native_before = adapter._source.get_state()
        model_fields = ("body_mass", "body_inertia", "body_invweight0", "dof_invweight0", "geom_pos")
        model_before = [{name: getattr(model, name).copy() for name in model_fields} for model in adapter._models]
        bad = before.clone()
        if fault == "episode_target_overflow":
            bad["task", "target_pos"][2, 0] = 1e300
        else:
            # A valid change in lane zero precedes the invalid later lane. The
            # native variation routine writes models before validating masses,
            # so unchanged public columns alone cannot prove atomic failure.
            bad["physics", "agent", "arm_density"][0] = 1200
            bad["physics", "agent", "arm_density"][2] = -1 if fault == "negative_physics" else 1e300
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(bad)
        assert_tree(adapter.get_state(), before, exact=True)
        assert_tree(adapter._source.get_state(), native_before, exact=True)
        for model, fields in zip(adapter._models, model_before, strict=True):
            for name, value in fields.items():
                np.testing.assert_array_equal(getattr(model, name), value, err_msg=name)
    finally:
        adapter.close()


@pytest.mark.parametrize("invalid_density", [-1, 1e300])
def test_invalid_later_lane_configuration_preserves_models_and_checkpoint(adapter_type, invalid_density):
    adapter = adapter_type("reacher", 3, "cpu", 5)
    try:
        before = adapter.get_state()
        native_before = adapter._source.get_state()
        masses = [model.body_mass.copy() for model in adapter._models]
        with pytest.raises(ValueError):
            adapter.configure_physics({"agent.arm_density": [[1200], [1000], [invalid_density]]})
        assert_tree(adapter.get_state(), before, exact=True)
        assert_tree(adapter._source.get_state(), native_before, exact=True)
        for model, expected in zip(adapter._models, masses, strict=True):
            np.testing.assert_array_equal(model.body_mass, expected)
    finally:
        adapter.close()


def test_prepare_restore_owns_the_checkpoint_until_commit(adapter_type):
    adapter = adapter_type("reacher", 2, "cpu", 7)
    try:
        initial = adapter.get_state()
        checkpoint = adapter.reset(31)
        held = checkpoint.clone()
        adapter.set_state(initial)
        adapter.prepare_restore(checkpoint)
        checkpoint["qpos"].zero_()
        assert_tree(adapter.get_state(), initial, exact=True)
        adapter.commit_restore()
        assert_tree(adapter.get_state(), held, exact=True)
    finally:
        adapter.close()
