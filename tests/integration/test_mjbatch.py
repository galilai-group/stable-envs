"""The shared batched MuJoCo backend: native mujoco.batch on CPU, MuJoCo Warp on CUDA.

Covers routing, bit-exactness of the native backend with one MjData per environment, writes to a subset of
environments, per-environment model parameters, the MuJoCo Warp trajectory against native physics, buffer
overflow errors and the parallel reset pool.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco.batch")

import mujoco

from stable_envs._execution import mjbatch
from stable_envs._execution.mjbatch import (
    BufferOverflowError,
    NativeBatch,
    ResetPool,
    WarpBatch,
    make_backend,
    resolve_backend,
)

XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="upper" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.04" mass="1"/>
      <body name="lower" pos="0.4 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="0.5"/>
        <site name="tip" pos="0.4 0 0"/>
      </body>
    </body>
    <body name="box" pos="0.6 0.5 0.1">
      <freejoint/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="0.3"/>
    </body>
    <body name="target" mocap="true" pos="0 0 0.5">
      <geom type="sphere" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="5" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="elbow" gear="3" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
  <sensor>
    <framepos objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""
B, STEPS, FRAME_SKIP = 6, 200, 4


def _cuda_problem() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    pytest.importorskip("mujoco_warp")
    from stable_envs._core.warp_cuda import warp_cuda_problem

    return warp_cuda_problem()


def cuda(fn):
    problem = _cuda_problem() if torch.cuda.is_available() else "CUDA required"
    return pytest.mark.gpu(pytest.mark.skipif(problem is not None, reason=str(problem))(fn))


def _model(mass=None):
    model = mujoco.MjModel.from_xml_string(XML)
    if mass is not None:
        model.body_mass[model.body("box").id] = mass
    return model


def _ctrl(t, n=B):
    """Smooth controls, different per environment."""
    phase = np.arange(n)[:, None] * 0.7 + np.array([0.0, 1.3])
    return np.sin(0.05 * t + phase) * 0.8


def _mocap(t, n=B):
    pos = np.zeros((n, 1, 3))
    pos[:, 0, 0] = 0.1 * np.sin(0.02 * t + np.arange(n))
    pos[:, 0, 2] = 0.5
    return pos


def test_routing_follows_device_and_cutoff():
    assert resolve_backend(1, "cpu") == "native"
    assert resolve_backend(4096, "cpu", "auto") == "native"
    assert resolve_backend(1, "cuda") == "mjwarp"
    assert resolve_backend(1, "cuda", None) == "mjwarp"
    assert resolve_backend(255, "cuda", "auto", 256) == "native"
    assert resolve_backend(256, "cuda", "auto", 256) == "mjwarp"
    assert resolve_backend(4096, "cuda", "native") == "native"
    assert resolve_backend(1, "cuda", "mjwarp") == "mjwarp"
    with pytest.raises(ValueError, match="requires device='cuda'"):
        resolve_backend(1, "cpu", "mjwarp")
    with pytest.raises(ValueError, match="simulation_backend must be"):
        resolve_backend(1, "cpu", "mjx")
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="mjwarp_min_envs must be an int >= 1"):
            resolve_backend(1, "cpu", "auto", bad)


def test_native_threads_follow_the_affinity(monkeypatch):
    monkeypatch.setattr(mjbatch, "cpu_affinity", lambda: [0, 1])
    assert mjbatch.native_threads(64) == 2
    monkeypatch.setattr(mjbatch, "cpu_affinity", lambda: list(range(32)))
    assert mjbatch.native_threads(64) == 8
    assert mjbatch.native_threads(3) == 3


def _reference_rollout(models, steps, frame_skip, rne_post):
    datas = [mujoco.MjData(m) for m in models]
    for m, d in zip(models, datas):
        mujoco.mj_forward(m, d)
    out = []
    for t in range(steps):
        ctrl, mocap = _ctrl(t, len(models)), _mocap(t, len(models))
        for i, (m, d) in enumerate(zip(models, datas)):
            d.ctrl[:] = ctrl[i]
            d.mocap_pos[:] = mocap[i]
            mujoco.mj_step(m, d, nstep=frame_skip)
            if rne_post:
                mujoco.mj_rnePostConstraint(m, d)
        names = ("qpos", "qvel", "qacc_warmstart", "site_xpos", "sensordata", "cfrc_ext")
        out.append({name: np.stack([getattr(d, name).copy() for d in datas]) for name in names})
    return out


@pytest.mark.parametrize("rne_post", [False, True])
def test_native_backend_matches_one_mjdata_per_env_bit_for_bit(rne_post):
    models = [_model(0.3 + 0.1 * i) for i in range(B)]
    want = _reference_rollout(models, 40, FRAME_SKIP, rne_post)
    outputs = ("site_xpos", "sensordata", "cfrc_ext")
    with NativeBatch(models, B, frame_skip=FRAME_SKIP, rne_postconstraint=rne_post, outputs=outputs) as batch:
        for t, expected in enumerate(want):
            batch.step(torch.as_tensor(_ctrl(t)), mocap_pos=_mocap(t))
            for name, value in expected.items():
                if name == "cfrc_ext" and not rne_post:
                    continue
                got = batch.data(name).numpy()
                assert got.dtype == np.float64
                np.testing.assert_array_equal(got, value, err_msg=f"{name} at step {t}")


def test_native_set_physics_writes_only_the_given_envs():
    with NativeBatch(_model(), B, frame_skip=FRAME_SKIP) as batch, NativeBatch(_model(), B) as other:
        for t in range(10):
            batch.step(_ctrl(t))
        before = batch.get_physics()
        assert set(before) == set(mjbatch.PHYSICS_FIELDS)
        assert before["qpos"].dtype == torch.float64 and before["time"].shape == (B,)
        fresh = other.get_physics([1, 4])
        batch.set_physics(fresh, env_ids=[1, 4])
        after = batch.get_physics()
        for name in before:
            keep = [0, 2, 3, 5]
            assert torch.equal(after[name][keep], before[name][keep]), name
            assert torch.equal(after[name][[1, 4]], fresh[name]), name
        # Restoring a snapshot replays the trajectory bit for bit.
        batch.set_physics(before)
        batch.step(_ctrl(10))
        first = batch.get_physics()
        batch.set_physics(before)
        batch.step(_ctrl(10))
        assert all(torch.equal(first[k], v) for k, v in batch.get_physics().items())
        batch.reset([0])
        assert torch.equal(batch.data("qpos")[0], torch.as_tensor(_model().qpos0))


def test_native_unstable_envs_report_diverged_envs_until_restored():
    with NativeBatch(_model(), 3) as batch:
        start = batch.get_physics()
        bad = {k: v.clone() for k, v in start.items()}
        bad["qvel"][1] = 1e12
        batch.set_physics(bad)
        batch.step(np.zeros((3, 2)))
        assert batch.unstable_envs().tolist() == [1]
        batch.set_physics(start)
        assert batch.unstable_envs().size == 0


def test_backend_on_a_cuda_env_device_returns_cuda_copies():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with NativeBatch(_model(), 2, "cuda", outputs=["xpos"]) as batch:
        batch.step(torch.zeros(2, 2, device="cuda"))
        assert batch.get_physics()["qpos"].device.type == "cuda"
        assert batch.read(["xpos"])["xpos"].device.type == "cuda"


@cuda
def test_mjwarp_follows_native_physics_over_smooth_controls():
    models = [_model(0.3 + 0.1 * i) for i in range(B)]
    native = NativeBatch(models, B, frame_skip=FRAME_SKIP, outputs=["site_xpos"])
    warp = WarpBatch(models, B, "cuda", frame_skip=FRAME_SKIP)
    assert "body_mass" in warp._batched
    worst = {}
    for t in range(STEPS):
        for batch in (native, warp):
            batch.step(_ctrl(t), mocap_pos=_mocap(t))
        for name in ("qpos", "qvel", "site_xpos", "mocap_pos"):
            gap = (warp.data(name).double().cpu() - native.data(name)).abs().max().item()
            worst[name] = max(worst.get(name, 0.0), gap)
    warp.check_buffers()
    native.close()
    warp.close()
    print("worst deviation over", STEPS, "steps:", worst)
    assert worst["qpos"] < 2e-3 and worst["site_xpos"] < 2e-3 and worst["mocap_pos"] < 1e-6, worst


@cuda
def test_mjwarp_set_physics_and_reset_write_only_the_given_envs():
    with WarpBatch(_model(), B, "cuda", frame_skip=FRAME_SKIP) as batch:
        for t in range(10):
            batch.step(torch.as_tensor(_ctrl(t), device="cuda"))
        before = batch.get_physics()
        assert before["qpos"].device.type == "cuda" and before["qpos"].dtype == torch.float32
        fresh = {k: v[:2].clone() + 0.01 for k, v in before.items() if k in ("qpos", "qvel")}
        batch.set_physics(fresh, env_ids=torch.tensor([1, 4], device="cuda"))
        after = batch.get_physics()
        keep = [0, 2, 3, 5]
        for name in before:
            assert torch.equal(after[name][keep], before[name][keep]), name
        assert torch.equal(after["qpos"][[1, 4]], fresh["qpos"])
        batch.reset(env_ids=[2])
        assert torch.allclose(batch.data("qpos")[2].double().cpu(), torch.as_tensor(_model().qpos0))
        assert float(batch.data("time")[2]) == 0.0 and float(batch.data("time")[0]) > 0.0


def test_native_step_advances_only_active_envs():
    with (
        NativeBatch(_model(), B, frame_skip=FRAME_SKIP, outputs=["xpos"]) as batch,
        NativeBatch(_model(), B, frame_skip=FRAME_SKIP, outputs=["xpos"]) as full,
    ):
        for t in range(3):
            batch.step(_ctrl(t))
            full.step(_ctrl(t))
        before = batch.get_physics()
        mask = np.array([True, False, True, False, False, True])
        batch.step(_ctrl(3), active=mask)
        full.step(_ctrl(3))
        after, want = batch.get_physics(), full.get_physics()
        for name in before:
            assert torch.equal(after[name][~mask], before[name][~mask]), name
            assert torch.equal(after[name][mask], want[name][mask]), name
        assert torch.equal(batch.data("xpos")[mask], full.data("xpos")[mask])


@cuda
def test_mjwarp_step_keeps_inactive_envs():
    with WarpBatch(_model(), B, "cuda", frame_skip=FRAME_SKIP, outputs=["site_xpos"]) as batch:
        for t in range(3):
            batch.step(_ctrl(t))
        before = batch.read([*batch.fields, "site_xpos"])
        active = torch.tensor([True, False, True, False, False, True], device="cuda")
        batch.step(_ctrl(3), active=active)
        after = batch.read([*batch.fields, "site_xpos"])
        for name in before:
            assert torch.equal(after[name][~active], before[name][~active]), name
        assert not torch.equal(after["qpos"][active], before["qpos"][active])


@cuda
def test_mjwarp_buffer_overflow_raises_a_clear_error():
    model = mujoco.MjModel.from_xml_string(XML.replace('pos="0.6 0.5 0.1"', 'pos="0.6 0.5 0.3"'))
    with pytest.raises(BufferOverflowError, match="naconmax"):
        WarpBatch(_model(), 8, "cuda", naconmax=8)
    with WarpBatch(model, 8, "cuda", naconmax=8) as batch:
        # The boxes fall onto the floor, and the batch then needs more than 8 contacts. The step reports the
        # overflow without a host synchronization, so it can come one or two steps late.
        with pytest.raises(BufferOverflowError, match="naconmax=8"):
            for t in range(100):
                batch.step(np.zeros((8, 2)))
            batch.check_buffers()


@cuda
def test_mjwarp_sensors_hold_the_values_of_the_last_substep():
    # Only the last substep computes sensors; the framepos sensor then equals the site position of that substep.
    with WarpBatch(_model(), B, "cuda", frame_skip=FRAME_SKIP, outputs=["site_xpos"]) as batch:
        for t in range(5):
            batch.step(_ctrl(t))
            assert torch.equal(batch.data("sensordata"), batch.data("site_xpos")[:, 0])


@cuda
def test_mjwarp_excluded_pairs_make_no_contacts():
    model = _model()
    pair = (model.geom("floor").id, model.geom("box").id)
    box_z = model.jnt_qposadr[model.body("box").jntadr[0]] + 2
    with WarpBatch(model, 2, "cuda") as kept, WarpBatch(model, 2, "cuda", exclude_pairs=[pair]) as dropped:
        assert len(dropped._m.nxn_geom_pair_filtered) == len(kept._m.nxn_geom_pair_filtered) - 1
        for _ in range(100):
            kept.step(np.zeros((2, 2)))
            dropped.step(np.zeros((2, 2)))
        # The box rests on the floor, or falls through it when the pair is excluded.
        assert kept.data("qpos")[:, box_z].min() > 0.05
        assert dropped.data("qpos")[:, box_z].max() < -0.5


@cuda
def test_make_backend_builds_the_routed_backend():
    backend = resolve_backend(4, "cuda", "auto", 8)
    with make_backend(backend, _model(), 4, "cuda") as batch:
        assert isinstance(batch, NativeBatch)
    with make_backend(resolve_backend(4, "cuda"), _model(), 4, "cuda") as batch:
        assert isinstance(batch, WarpBatch)


def _source_factory():
    return mujoco.MjData(_model())


def _source_reset(data, env_id, seed):
    model = _model()
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    data.qpos[:2] = rng.uniform(-0.5, 0.5, 2)
    mujoco.mj_forward(model, data)
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "env": np.int64(env_id)}


@pytest.mark.parametrize("owners,processes", [(1, False), (3, False), (2, True)])
def test_reset_pool_returns_rows_in_env_order(owners, processes):
    ids, seeds = [5, 0, 3, 2], [50, 0, 30, 20]
    pool = ResetPool(_source_factory, _source_reset, owners, processes=processes)
    try:
        values = pool.run(ids, seeds)
    finally:
        pool.close()
    assert values["env"].tolist() == ids
    want = np.stack([_source_reset(_source_factory(), i, s)["qpos"] for i, s in zip(ids, seeds)])
    np.testing.assert_array_equal(values["qpos"], want)
    with NativeBatch(_model(), 6) as batch:
        batch.set_physics({k: values[k] for k in ("qpos", "qvel")}, env_ids=ids)
        np.testing.assert_array_equal(batch.data("qpos").numpy()[ids], want)
