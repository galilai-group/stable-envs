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
