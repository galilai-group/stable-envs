"""Small independent MuJoCo models exercise the Warp execution boundary."""

from contextlib import closing

import numpy as np
import pytest
import torch
from tensordict import TensorDict

mujoco = pytest.importorskip("mujoco")
mjw = pytest.importorskip("mujoco_warp")

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


def test_per_world_model_reload_and_fixed_geometry_refresh():
    models = [model(), model()]
    models[1].opt.gravity[:] = [2, 0, -4]
    models[1].geom_pos[0] = [0.5, 0.4, 0.3]
    models[1].geom_quat[0] = [np.cos(0.2), 0, np.sin(0.2), 0]
    state, natives = checkpoint(models)
    with closing(WarpRuntime(models, "cpu", 1)) as runtime:
        runtime.load(state)
        expected = np.stack([d.geom_xpos[0] for d in natives])
        np.testing.assert_allclose(runtime.snapshot().data("geom_xpos")[:, 0], expected)
        assert runtime.snapshot().data("xmat").shape == (2, models[0].nbody, 9)
        assert runtime.snapshot().model("site_size").shape == (2, 1, 3)
        before = runtime.physical_state()
        runtime.advance(torch.zeros(2, 1))
        assert not torch.equal(runtime.physical_state()["qvel"][0], runtime.physical_state()["qvel"][1])
        assert torch.equal(before["qvel"], state["qvel"].float())
        models[1].geom_pos[0] = [-0.4, 0.1, 0.7]
        runtime.load(state, models=models)
        np.testing.assert_allclose(runtime.snapshot().data("geom_xpos")[1, 0], models[1].geom_pos[0])
        np.testing.assert_allclose(runtime.snapshot().model("opt.gravity")[1], models[1].opt.gravity)


def test_activation_checkpoint_is_loaded_and_integrated():
    xml = XML.format(integrator="Euler").replace(
        '<motor joint="hinge"/>', '<general joint="hinge" dyntype="filter" dynprm=".04" gainprm="1"/>'
    )
    m = mujoco.MjModel.from_xml_string(xml)
    state, (native,) = checkpoint([m])
    native.act[:] = 0.25
    state["activation"] = torch.tensor([[0.25]], dtype=torch.float64)
    mujoco.mj_forward(m, native)
    with closing(WarpRuntime([m], "cpu", 3)) as runtime:
        runtime.load(state)
        assert torch.equal(runtime.physical_state()["activation"], state["activation"].float())
        runtime.advance(torch.tensor([[0.7]]))
        source_round(m, native, 0.7, 3)
        tol = 64 * np.finfo(np.float32).eps
        np.testing.assert_allclose(runtime.physical_state()["activation"][0], native.act, rtol=tol, atol=tol)
        np.testing.assert_allclose(runtime.physical_state()["qvel"][0], native.qvel, rtol=tol, atol=tol)


def test_restore_preserves_warmstart_and_validates_before_mutation():
    m = model()
    state, _ = checkpoint([m])
    state["warmstart"].fill_(3.125)
    with closing(WarpRuntime([m], "cpu", 1)) as runtime:
        runtime.load(state)
        before = runtime.physical_state()
        assert torch.equal(before["warmstart"], state["warmstart"].float())
        invalid = state.clone()
        invalid["qvel"].fill_(float("nan"))
        with pytest.raises(ValueError, match="Nonfinite"):
            runtime.load(invalid)
        invalid["qvel"].fill_(1e300)
        with pytest.raises(ValueError, match="float32"):
            runtime.load(invalid)
        assert torch.equal(runtime.physical_state()["qpos"], before["qpos"])
        assert torch.equal(runtime.physical_state()["warmstart"], before["warmstart"])
        m.opt.gravity[:] = [1e300, 0, -9.81]
        with pytest.raises(ValueError, match="opt.gravity"):
            runtime.load(state, models=[m])
        np.testing.assert_array_equal(runtime.snapshot().model("opt.gravity"), [[0, 0, np.float32(-9.81)]])
        m.opt.gravity[:] = [0, 0, -9.81]
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        with pytest.raises(ValueError, match="integrator"):
            runtime.load(state, models=[m])


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
                runtime.snapshot().model("stat.meaninertia"),
                torch.tensor([m.stat.meaninertia for m in models], dtype=torch.float32),
            )
            tol = 8 * np.finfo(np.float32).eps
            for name in ("geom_xpos", "geom_xmat", "site_xpos"):
                expected = np.stack([getattr(env.physics.data, name) for env in native.envs])
                np.testing.assert_allclose(runtime.snapshot().data(name), expected, rtol=tol, atol=tol)


def test_capacity_and_nonfinite_faults_raise():
    m = model()
    state, _ = checkpoint([m])
    with closing(WarpRuntime([m], "cpu", 1)) as runtime:
        runtime.load(state)
        runtime._data_view("overflow").fill_(1)
        with pytest.raises(RuntimeError, match="capacity overflow"):
            runtime._check_result()
        runtime._data_view("overflow").fill_(int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS))
        runtime._check_result()
        runtime._data_view("overflow").zero_()
        runtime._data_view("qpos").fill_(float("nan"))
        with pytest.raises(RuntimeError, match="nonfinite qpos"):
            runtime._check_result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime interop")
def test_cuda_capture_restore_and_alternate_caller_streams():
    m = model()
    state, (native,) = checkpoint([m])
    stream = torch.cuda.Stream()
    with closing(WarpRuntime([m], "cuda", 3)) as runtime:
        runtime.load(state)
        with torch.cuda.stream(stream):
            runtime.advance(torch.tensor([[0.3]], device="cuda"))
            first = runtime.physical_state()
        runtime.load(state)
        runtime.advance(torch.tensor([[0.3]], device="cuda"))
        assert torch.equal(first["qpos"], runtime.physical_state()["qpos"])
        assert torch.equal(first["qvel"], runtime.physical_state()["qvel"])
        source_round(m, native, 0.3, 3)
        with torch.cuda.stream(stream):
            runtime.advance(torch.tensor([[-0.2]], device="cuda"))
        source_round(m, native, -0.2, 3)
        tol = 64 * np.finfo(np.float32).eps
        np.testing.assert_allclose(runtime.physical_state()["qpos"].cpu()[0], native.qpos, rtol=tol, atol=tol)
        np.testing.assert_allclose(runtime.physical_state()["qvel"].cpu()[0], native.qvel, rtol=tol, atol=tol)
        np.testing.assert_allclose(
            runtime.snapshot().data("sensordata").cpu()[0], native.sensordata, rtol=tol, atol=tol
        )
