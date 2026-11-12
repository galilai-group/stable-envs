"""DMC's legacy step order on the shared MuJoCo Warp batch, and per-environment model values."""

from contextlib import closing

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]

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


def source_round(m, d, control, substeps):
    """dm_control's ``Physics.step`` with ``legacy_step``."""
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
def test_round_uses_source_integrator_and_keeps_acceleration_sensors(integrator, substeps):
    from stable_envs._execution.mjbatch import WarpBatch
    from stable_envs.envs.dmcontrol.simulation.mjwarp import LegacyStep

    m = mujoco.MjModel.from_xml_string(XML.format(integrator=integrator))
    native = mujoco.MjData(m)
    native.qpos[:], native.qvel[:] = 0.4, 0.2
    mujoco.mj_forward(m, native)
    step = LegacyStep(m, 1, substeps, "cuda")
    with closing(WarpBatch(m, 1, "cuda", frame_skip=substeps, step_fn=step)) as batch:
        batch.set_physics({"qpos": [[0.4]], "qvel": [[0.2]]})
        batch.step(torch.tensor([[0.3]]))
        source_round(m, native, 0.3, substeps)
        # One smooth round with no contacts: a float32-sized bound separates integrator and sensor-stage errors.
        tol = 64 * np.finfo(np.float32).eps
        for name in ("qpos", "qvel", "sensordata"):
            np.testing.assert_allclose(batch.data(name).cpu().numpy()[0], getattr(native, name), atol=tol, rtol=tol)
        assert np.linalg.norm(native.sensordata[:3]) > 1
        assert int(step.flag[0]) == 0


@pytest.mark.parametrize("domain", ["ball_in_cup", "manipulator", "walker"])
def test_model_variations_and_episode_targets_are_per_environment(domain):
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol.simulation.factory import make_local
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
    from stable_envs.envs.dmcontrol.simulation.physics import LEAVES

    with closing(make_local(domain, 2, "cpu", 19)) as native, closing(DMCWarpAdapter(domain, 2, "cuda", 19)) as gpu:
        # Ball radius changes collision bounds and mass; Walker tilts a fixed plane; Manipulator draws a fixed
        # target body's position and rotation on each reset.
        values = {leaf.path: [[leaf.low], [leaf.high]] for leaf in LEAVES[domain]}
        native.configure_physics(values)
        gpu.configure_physics(values)
        native.reset(31)
        gpu.reset(31)
        models = [env.physics.model.ptr for env in native.envs]
        for name in ("geom_size", "geom_rbound", "body_mass", "body_inertia", "body_invweight0", "dof_invweight0"):
            expected = torch.tensor(np.stack([getattr(m, name) for m in models]), dtype=torch.float32)
            assert torch.equal(gpu.batch.model_field(name).cpu().reshape(expected.shape), expected), name
        expected = torch.tensor(np.stack([m.opt.gravity for m in models]), dtype=torch.float32)
        assert torch.equal(gpu.batch.model_field("opt.gravity").cpu(), expected)
        tol = 8 * np.finfo(np.float32).eps
        for name in ("geom_xpos", "geom_xmat", "site_xpos"):
            expected = np.stack([getattr(env.physics.data, name) for env in native.envs])
            actual = gpu.batch.data(name).cpu().numpy().reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, err_msg=name)
