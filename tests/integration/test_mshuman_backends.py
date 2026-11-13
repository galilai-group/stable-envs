"""MSHumanLocomotion steps CPU lanes in one native mujoco.batch and CUDA lanes on MuJoCo Warp by default.

Covers backend dispatch, float32 State at the API boundary, fidelity against single-MjData MuJoCo,
State restore and replay, action clamping, and release of the batch's native threads.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco.batch")
pytest.importorskip("mujoco_warp")

import mujoco

import stable_envs as se
from stable_envs._execution import mjbatch
from stable_envs.envs.mshuman.assets import model
from stable_envs.envs.mshuman.simulation.adapter import MSHumanBatchAdapter, MSHumanWarpAdapter
from tests.helpers.reference.semantics import assert_same_state

LANES, STEPS, SEED = 4, 12, 7
FIELDS = ("qpos", "qvel", "act", "time", "warmstart")


def _cuda_problem() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    from stable_envs._core.warp_cuda import warp_cuda_problem

    return warp_cuda_problem()


_CUDA = _cuda_problem()
cuda_only = [pytest.mark.gpu, pytest.mark.skipif(_CUDA is not None, reason=str(_CUDA))]
DEVICES = ["cpu", pytest.param("cuda", marks=cuda_only), pytest.param("cuda/mjwarp", marks=cuda_only)]


def _make(lanes=LANES, device="cpu"):
    """``"cuda/mjwarp"`` selects MuJoCo Warp; a plain device uses the automatic backend."""
    device, _, backend = device.partition("/")
    return se.make("MSHumanLocomotion", num_envs=lanes, device=device, simulation_backend=backend or "auto")


def _actions(env, steps=STEPS, seed=SEED):
    rng = np.random.default_rng(seed)
    return [torch.as_tensor(env.action_spec.sample(env.num_envs, rng)).cpu().numpy() for _ in range(steps)]


def _round(d):
    """Round one MjData's integration State to float32, as the native batch does at every step boundary."""
    for name in ("qpos", "qvel", "act", "qacc_warmstart"):
        getattr(d, name)[:] = getattr(d, name).astype(np.float32)
    d.time = float(np.float32(d.time))


def _single_env_rollout(actions):
    """Step one MjData per lane with mj_step from the ``init`` keyframe, rounding as the native batch does."""
    m = model()
    lanes = [mujoco.MjData(m) for _ in range(len(actions[0]))]
    for d in lanes:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        _round(d)
    out = []
    for action in actions:
        for d, ctrl in zip(lanes, np.asarray(action, np.float32)):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
            _round(d)
        values = {
            key: np.stack([np.atleast_1d(getattr(d, name)).copy() for d in lanes])
            for key, name in zip(FIELDS, ("qpos", "qvel", "act", "time", "qacc_warmstart"))
        }
        out.append({key: torch.from_numpy(value.reshape(len(lanes), -1)) for key, value in values.items()})
    return out


def _threads() -> int:
    return len(os.listdir("/proc/self/task"))


def test_cpu_env_steps_on_the_native_batch():
    with _make() as env:
        assert type(env._simulator) is MSHumanBatchAdapter
    with se.make("MSHumanLocomotion", num_envs=LANES, device="cpu") as env:
        assert env.simulation_backend == "native"


@pytest.mark.parametrize("device", [pytest.param("cuda", marks=cuda_only)])
def test_cuda_env_steps_on_mujoco_warp_by_default_at_every_batch_size(device):
    for lanes in (1, LANES):
        with se.make("MSHumanLocomotion", num_envs=lanes, device=device) as env:
            assert env.simulation_backend == "mjwarp"
            assert type(env._simulator) is MSHumanWarpAdapter


@pytest.mark.parametrize("device", [pytest.param("cuda", marks=cuda_only)])
def test_small_cuda_env_steps_on_the_native_batch(device):
    with _make(device=device) as env:
        assert env.simulation_backend == "native"
        assert type(env._simulator) is MSHumanBatchAdapter


@pytest.mark.parametrize("device", [pytest.param("cuda", marks=cuda_only)])
def test_cuda_env_steps_on_mujoco_warp_from_the_cutoff(device):
    with se.make("MSHumanLocomotion", num_envs=2, device=device, simulation_backend="auto", mjwarp_min_envs=2) as env:
        assert env.simulation_backend == "mjwarp"
        assert type(env._simulator) is MSHumanWarpAdapter
        assert env._simulator.batch._step_graph is not None
    with se.make("MSHumanLocomotion", num_envs=1, device=device, simulation_backend="mjwarp") as env:
        assert type(env._simulator) is MSHumanWarpAdapter


def test_mjwarp_backend_requires_cuda():
    with pytest.raises(ValueError, match="requires device='cuda'"):
        se.make("MSHumanLocomotion", num_envs=1, device="cpu", simulation_backend="mjwarp")


@pytest.mark.parametrize("device", DEVICES)
def test_state_and_step_outputs_are_float32_on_the_env_device(device):
    with _make(device=device) as env:
        assert env.action_spec.dtype == torch.float32
        for state in (env.reset(seed=SEED), env.get_state()):
            assert set(state.keys()) == set(FIELDS)
            for key in FIELDS:
                assert state[key].dtype == torch.float32 and state[key].device.type == env.device.type, key
        out = env.step(_actions(env, steps=1)[0], frames=None)
        for key, value in out.items(True, True):
            assert value.dtype in (torch.float32, torch.bool), (key, value.dtype)
            assert value.device.type == env.device.type, key


@pytest.mark.parametrize("device", DEVICES)
def test_action_and_state_dtypes_are_cast_at_the_boundary(device):
    with _make(device=device) as env:
        start = env.reset()
        env.set_state(start.apply(lambda v: v.double()))
        assert_same_state(env.get_state(), start)
        if device != "cpu":
            return
        action = _actions(env, steps=1)[0]
        want = env.step(np.asarray(action, np.float64), frames=None)["state"]
        env.reset()
        assert_same_state(env.step(torch.as_tensor(action, dtype=torch.float32), frames=None)["state"], want)


@pytest.mark.parametrize("device", DEVICES)
def test_step_snapshots_stack_as_the_core_does(device):
    from stable_envs._core.stepping import _stack_states

    with _make(device=device) as env:
        env.reset()
        values = [env.step(action, frames=None)["state"] for action in _actions(env, steps=3)]
        assert_same_state(env._stack_states(values), _stack_states(values))


def test_native_batch_matches_single_env_mujoco_with_float32_rounding_bit_for_bit():
    with _make(16) as env:
        actions = _actions(env)
        env.reset(seed=SEED)
        for action, want in zip(actions, _single_env_rollout(actions)):
            state = env.step(action, frames=None)["state"]
            for key in FIELDS:
                assert torch.equal(state[key], want[key].to(torch.float32)), key


def test_native_batch_results_do_not_depend_on_the_thread_count(monkeypatch):
    lanes = 8
    actions = np.random.default_rng(SEED).uniform(0, 1, (6, lanes, 100)).astype(np.float32)
    runs = []
    for affinity in ([0], list(range(lanes))):
        monkeypatch.setattr(mjbatch, "cpu_limit", lambda affinity=affinity: len(affinity))
        with _make(lanes) as env:
            assert env._simulator.b.nthread == len(affinity)
            env.reset()
            runs.append([env.step(a, frames=None)["state"] for a in actions])
    for one, many in zip(*runs):
        assert_same_state(many, one)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_is_deterministic(device):
    with _make(device=device) as env, _make(device=device) as other:
        first = env.reset(seed=SEED)
        assert_same_state(other.reset(seed=SEED), first)
        for action in _actions(env, steps=3):
            env.step(action, frames=None)
        assert_same_state(env.reset(seed=SEED), first)
        assert torch.equal(first["time"], torch.zeros_like(first["time"]))


@pytest.mark.parametrize("device", DEVICES)
def test_restored_state_replays_bit_for_bit(device):
    with _make(device=device) as env, _make(device=device) as other:
        actions = _actions(env)
        env.reset(seed=SEED)
        for action in actions[:5]:
            env.step(action, frames=None)
        saved = env.get_state()
        expected = [env.step(action, frames=None)["state"] for action in actions[5:]]
        env.set_state(saved)
        assert_same_state(env.get_state(), saved)
        if device != "cpu":
            return
        for action, want in zip(actions[5:], expected):
            assert_same_state(env.step(action, frames=None)["state"], want)
        other.set_state(saved)
        sequence = torch.as_tensor(np.stack(actions[5:], axis=1))
        out = other.step(sequence, frames=None)
        for t, want in enumerate(expected):
            assert_same_state(out["states"][:, t], want)


def test_actions_are_validated_and_clamped_to_ctrlrange(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with _make() as env:
        with pytest.raises(ValueError, match=r"actions must be \[B, 100\]"):
            env.step(np.zeros((LANES, 99)))
        base = np.asarray(_actions(env, steps=1)[0], np.float32)
        for special, bound in ((np.nan, 0.0), (np.inf, 1.0), (-np.inf, 0.0), (1e11, 1.0), (-1e11, 0.0), (3.0, 1.0)):
            bad, clamped = base.copy(), base.copy()
            bad[:, ::2], clamped[:, ::2] = special, bound
            env.reset()
            got = env.step(bad, frames=None)["state"]
            env.reset()
            assert_same_state(got, env.step(clamped, frames=None)["state"])
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()


def test_a_diverging_lane_raises_until_the_state_is_restored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handler = mujoco.get_mju_user_warning()
    # Importing dm_control installs its own handler, which would hide MuJoCo's default one.
    mujoco.set_mju_user_warning(None)
    try:
        with _make() as env:
            start = env.reset()
            action = _actions(env, steps=1)[0]
            bad = start.clone()
            bad["qvel"][1] = 1e12
            env.set_state(bad)
            for _ in range(2):
                with pytest.raises(RuntimeError, match=r"Physics state is invalid in lanes \[1\]"):
                    env.step(action, frames=None)
            env.set_state(start)
            env.step(action, frames=None)
    finally:
        mujoco.set_mju_user_warning(handler)
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()


def test_close_is_idempotent_and_releases_the_batch():
    adapter = MSHumanBatchAdapter(2, "cpu", model())
    adapter.close()
    adapter.close()
    assert adapter.b is None
    with pytest.raises(RuntimeError, match="MSHumanBatchAdapter is closed"):
        adapter.get_state()


def test_native_batch_threads_respect_the_cpu_limit(monkeypatch):
    monkeypatch.setattr(mjbatch, "cpu_limit", lambda: 2)
    with _make(64) as env:
        assert env._simulator.b.nthread == 2
    monkeypatch.setattr(mjbatch, "cpu_limit", lambda: 16)
    with _make(9) as env:
        assert env._simulator.b.nthread == 8


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counts threads in /proc/self/task")
def test_close_and_del_join_the_native_threads(monkeypatch):
    monkeypatch.setattr(mjbatch, "cpu_limit", lambda: 4)
    with _make(8) as env:
        env.step(np.zeros((8, 100)), frames=None)
    gc.collect()
    before = _threads()
    env = _make(8)
    assert env._simulator.b.nthread == 4
    assert _threads() == before + 3
    env.step(np.zeros((8, 100)), frames=None)
    env.close()
    gc.collect()
    assert _threads() == before
    env = _make(8)
    assert _threads() == before + 3
    del env
    gc.collect()
    assert _threads() == before
