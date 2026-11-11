"""Small independent MuJoCo models exercise the Warp execution boundary."""

from contextlib import closing

import numpy as np
import pytest
import torch
from tensordict import TensorDict

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import WarpRuntime  # noqa: E402

XML = """<mujoco>
  <option timestep="0.005" integrator="{integrator}"/>
  <worldbody>
    <geom name="target" pos="1 0 0" size=".03" contype="0" conaffinity="0"/>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" axis="0 1 0" damping=".1"/>
      <geom size=".03" fromto="0 0 0 0 0 -.5" type="capsule"/>
      <site name="tip" pos="0 0 -.5" size=".02"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
  <sensor><accelerometer site="tip"/><velocimeter site="tip"/></sensor>
</mujoco>"""


def model(integrator="Euler"):
    return mujoco.MjModel.from_xml_string(XML.format(integrator=integrator))


def checkpoint(models):
    datas = [mujoco.MjData(m) for m in models]
    for m, d in zip(models, datas, strict=True):
        d.qpos[:] = 0.4
        d.qvel[:] = 0.2
        mujoco.mj_forward(m, d)
    state = TensorDict(
        {
            "qpos": torch.from_numpy(np.stack([d.qpos for d in datas])),
            "qvel": torch.from_numpy(np.stack([d.qvel for d in datas])),
            "time": torch.zeros(len(models), 1, dtype=torch.float64),
            "warmstart": torch.from_numpy(np.stack([d.qacc_warmstart for d in datas])),
        },
        [len(models)],
    )
    return state, datas


def source_round(m, d, control, substeps):
    d.ctrl[:] = control
    if m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        mujoco.mj_step(m, d, substeps)
    else:
        mujoco.mj_step2(m, d)
        if substeps > 1:
            mujoco.mj_step(m, d, substeps - 1)
    mujoco.mj_step1(m, d)


@pytest.mark.parametrize(
    ("lookup_result", "trim_result", "message"),
    [(0, 0, None), (101, 0, r"resolve CUDA device 3.*error 101"), (0, 2, r"graph memory on device 3.*error 2")],
)
def test_graph_memory_trim_resolves_driver_device_and_reports_errors(lookup_result, trim_result, message, monkeypatch):
    import ctypes
    from types import SimpleNamespace

    from stable_envs.envs.dmcontrol.simulation import mjwarp_runtime

    calls = []

    def device_get(destination, ordinal):
        calls.append(("lookup", ordinal))
        # A CUDA driver device handle need not equal the caller's ordinal.
        ctypes.cast(destination, ctypes.POINTER(ctypes.c_int)).contents.value = 37
        return lookup_result

    def trim(device):
        calls.append(("trim", device))
        return trim_result

    driver = SimpleNamespace(cuDeviceGet=device_get, cuDeviceGraphMemTrim=trim)
    monkeypatch.setattr(mjwarp_runtime, "_cuda_graph_memory_driver", lambda: driver)
    if message is None:
        mjwarp_runtime._trim_cuda_graph_memory(3)
    else:
        with pytest.raises(RuntimeError, match=message):
            mjwarp_runtime._trim_cuda_graph_memory(3)
    assert calls == ([("lookup", 3)] if lookup_result else [("lookup", 3), ("trim", 37)])


def test_cpu_graph_memory_trim_does_not_load_cuda_driver(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation import mjwarp_runtime

    def unexpected_driver_load():
        pytest.fail("A CPU runtime tried to load the NVIDIA CUDA driver")

    monkeypatch.setattr(mjwarp_runtime, "_cuda_graph_memory_driver", unexpected_driver_load)
    with closing(WarpRuntime([model()], "cpu", 1)) as runtime:
        runtime.trim_graph_memory()


def test_retained_scratch_reuses_round_storage_without_aliasing_live_allocations(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp_pair import _RetainedScratch

    with closing(_RetainedScratch("cpu")) as scratch:
        sizes = [16, 48, 16]
        pointers = [scratch.allocate(size) for size in sizes]
        # Equal-sized allocations within a round still need independent storage.
        assert len(set(pointers)) == len(sizes)
        for index, buffer in enumerate(scratch.buffers):
            buffer.fill_(index + 1)
        for pointer, size in zip(pointers, sizes, strict=True):
            scratch.deallocate(pointer, size)
        assert [buffer[0].item() for buffer in scratch.buffers] == [1, 2, 3]

        def unexpected_allocation(*args, **kwargs):
            pytest.fail("A later round allocated new retained storage")

        monkeypatch.setattr(torch, "empty", unexpected_allocation)
        # The warm second round and both captured rounds reuse one sequence.
        for _ in range(3):
            scratch.begin_capture()
            assert [scratch.allocate(size) for size in sizes] == pointers
            scratch.finish_capture()
            assert scratch.sizes == sizes
            assert [buffer.data_ptr() for buffer in scratch.buffers] == pointers
            assert sum(buffer.numel() for buffer in scratch.buffers) == sum(sizes)


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        pytest.param([16, 32, 16], "changed a scratch allocation's size", id="changed-size"),
        pytest.param([48, 16, 16], "changed a scratch allocation's size", id="changed-order"),
        pytest.param([16, 48, 16, 8], "unexpected scratch storage", id="extra-allocation"),
        pytest.param([16, 48], "omitted a recorded scratch allocation", id="omitted-allocation"),
    ],
)
def test_retained_scratch_later_round_mismatch_fails_closed(requests, message, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp_pair import _RetainedScratch

    with closing(_RetainedScratch("cpu")) as scratch:
        sizes = [16, 48, 16]
        pointers = [scratch.allocate(size) for size in sizes]
        scratch.begin_capture()
        assert [scratch.allocate(size) for size in sizes] == pointers
        scratch.finish_capture()

        def unexpected_allocation(*args, **kwargs):
            pytest.fail("A mismatched round silently allocated replacement storage")

        monkeypatch.setattr(torch, "empty", unexpected_allocation)
        scratch.begin_capture()
        with pytest.raises(RuntimeError, match=message):
            for size in requests:
                scratch.allocate(size)
            scratch.finish_capture()
        assert scratch.sizes == sizes
        assert [buffer.data_ptr() for buffer in scratch.buffers] == pointers


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
@pytest.mark.parametrize("subset", [False, True])
def test_copy_physical_state_preserves_clock_unselected_lanes_and_ownership(device, subset):
    xml = XML.format(integrator="Euler").replace(
        '<motor joint="hinge"/>', '<general joint="hinge" dyntype="filter" dynprm="0.1"/>'
    )
    m = mujoco.MjModel.from_xml_string(xml)
    with closing(WarpRuntime([m, m], device, 1)) as runtime:
        with runtime._scope():
            for index, name in enumerate(runtime._physical.values()):
                view = runtime._data_view(name)
                view.copy_(torch.arange(view.numel(), device=device).reshape(view.shape) + (index + 1) / 7)
        source = runtime.physical_state()
        assert "activation" in source.keys()
        batch = 4 if subset else 2
        lanes = torch.tensor([3, 1], device=device) if subset else None
        destination = TensorDict(
            {
                key: torch.full((batch, value.shape[1]), -777.0, dtype=torch.float64, device=device)
                for key, value in source.items()
            },
            [batch],
            device=device,
        )
        # This clock value is deliberately not representable in float32.
        destination["time"].fill_(0.123456789012345)
        destination["untouched"] = torch.arange(batch, device=device)[:, None]
        expected = destination.clone()
        selected = lanes if subset else slice(None)
        for key, value in source.items():
            if key != "time":
                expected[key][selected] = value.double()
        runtime.copy_physical_state(destination, lanes=lanes, include_time=False)
        torch.testing.assert_close(destination, expected, atol=0, rtol=0)

        # Standalone callers can also copy the runtime clock explicitly.
        expected["time"][selected] = source["time"].double()
        runtime.copy_physical_state(destination, lanes=lanes)
        torch.testing.assert_close(destination, expected, atol=0, rtol=0)
        with runtime._scope():
            for name in runtime._physical.values():
                runtime._data_view(name).zero_()
        torch.testing.assert_close(destination, expected, atol=0, rtol=0)
        # Neither direction may borrow storage from the other.
        for key in source.keys():
            destination[key].fill_(123)
        for value in runtime.physical_state().values():
            assert torch.count_nonzero(value).item() == 0


@pytest.mark.parametrize("integrator", ["Euler", "RK4"])
@pytest.mark.parametrize("substeps", [1, 3])
def test_round_uses_source_integrator_and_retains_acceleration_sensors(integrator, substeps):
    m = model(integrator)
    state, (native,) = checkpoint([m])
    with closing(WarpRuntime([m], "cpu", substeps)) as runtime:
        runtime.load(state)
        runtime.advance(torch.tensor([[0.3]]))
        source_round(m, native, 0.3, substeps)
        # This smooth one-round calculation has no contacts. A float32-sized
        # error budget distinguishes integrator and sensor-stage mistakes.
        tol = 64 * np.finfo(np.float32).eps
        np.testing.assert_allclose(runtime.physical_state()["qpos"].numpy()[0], native.qpos, atol=tol, rtol=tol)
        np.testing.assert_allclose(runtime.physical_state()["qvel"].numpy()[0], native.qvel, atol=tol, rtol=tol)
        assert np.linalg.norm(native.sensordata[:3]) > 1
        np.testing.assert_allclose(
            runtime.snapshot().data("sensordata").numpy()[0], native.sensordata, atol=tol, rtol=tol
        )


@pytest.mark.parametrize("domain", ["ball_in_cup", "manipulator", "walker"])
def test_registered_model_variations_and_episode_targets_are_per_world(domain):
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol.simulation.factory import make_local
    from stable_envs.envs.dmcontrol.simulation.physics import LEAVES

    with closing(make_local(domain, 2, "cpu", 19)) as native:
        models = [env.physics.model.ptr for env in native.envs]
        with closing(WarpRuntime(models, "cpu", 1)) as runtime:
            # Reload an already allocated runtime: ball radius changes collision
            # bounds and mass; Walker tilts a fixed plane; Manipulator draws a
            # fixed target body's position and rotation on each reset.
            native.configure_physics({leaf.path: [[leaf.low], [leaf.high]] for leaf in LEAVES[domain]})
            native.reset(31)
            runtime.load(native.get_state(), models=models)
            for name in ("geom_size", "geom_rbound", "body_mass", "body_inertia", "body_invweight0", "dof_invweight0"):
                expected = torch.tensor(np.stack([getattr(m, name) for m in models]), dtype=torch.float32)
                assert torch.equal(runtime.snapshot().model(name), expected)
            assert torch.equal(
                runtime.snapshot().model("opt.gravity"),
                torch.tensor(np.stack([m.opt.gravity for m in models]), dtype=torch.float32),
            )
            tol = 8 * np.finfo(np.float32).eps
            for name in ("geom_xpos", "geom_xmat", "site_xpos"):
                expected = np.stack([getattr(env.physics.data, name) for env in native.envs])
                np.testing.assert_allclose(runtime.snapshot().data(name), expected, rtol=tol, atol=tol)
