"""Native regressions for the two corrected MuJoCo Warp 3.12 contact cases."""

from contextlib import closing

import numpy as np
import pytest
import torch
from tensordict import TensorDict

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import WarpRuntime  # noqa: E402

pytestmark = pytest.mark.family("dmc")
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact kernels")],
    ),
]


def _check_contact_step(models, datas, device):
    state = TensorDict(
        {
            "qpos": torch.tensor(np.stack([data.qpos for data in datas])),
            "qvel": torch.tensor(np.stack([data.qvel for data in datas])),
            "time": torch.zeros(len(models), 1, dtype=torch.float64),
            "warmstart": torch.tensor(np.stack([data.qacc_warmstart for data in datas])),
        },
        [len(models)],
    )
    assert all(data.ncon == 1 for data in datas)
    with closing(WarpRuntime(models, device, 1)) as runtime:
        runtime.load(state.to(device))
        count = int(runtime._d.nacon.numpy()[0])
        worlds = runtime._d.contact.worldid.numpy()[:count]
        assert sorted(worlds.tolist()) == list(range(len(models)))
        for index, world in enumerate(worlds):
            expected = datas[world].contact[0]
            for name in ("frame", "pos", "dist"):
                actual = getattr(runtime._d.contact, name).numpy()[index]
                np.testing.assert_allclose(
                    actual, np.asarray(getattr(expected, name)).reshape(actual.shape), atol=8e-7, rtol=8e-7
                )
        # Integrating the native constraint proves that contact geometry also
        # produces the correct impulse, rather than matching diagnostics only.
        for model, data in zip(models, datas, strict=True):
            mujoco.mj_step2(model, data)
            mujoco.mj_step1(model, data)
        runtime.advance(torch.empty(len(models), 0, device=device))
        physical = runtime.physical_state()
        np.testing.assert_allclose(
            physical["qpos"].cpu(), np.stack([data.qpos for data in datas]), atol=2e-6, rtol=2e-6
        )
        np.testing.assert_allclose(
            physical["qvel"].cpu(), np.stack([data.qvel for data in datas]), atol=4e-5, rtol=4e-5
        )


@pytest.mark.parametrize("device", DEVICES)
def test_plane_capsule_friction_frame_matches_native_per_world(device):
    models, datas = [], []
    for normal, axis in [([0, 0, 1], [0.2, 0.2, 1]), ([0.2, -0.1, 1], [0.4, 0.1, 1])]:
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        center = normal * (0.03 + 0.1 * abs(normal @ axis) - 0.005)
        fmt = lambda values: " ".join(map(str, values))
        model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep=".005" cone="pyramidal" iterations="100" tolerance="1e-8"/>
          <worldbody><geom type="plane" zaxis="{fmt(normal)}" size="1 1 .1"/>
            <body pos="{fmt(center)}"><freejoint/>
              <geom type="capsule" zaxis="{fmt(axis)}" size=".03 .1"/>
            </body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = [1, 0.7, -0.3, 0.1, 0.2, 0.3]
        mujoco.mj_forward(model, data)
        models.append(model)
        datas.append(data)
    _check_contact_step(models, datas, device)


@pytest.mark.parametrize("device", DEVICES)
def test_intersecting_capsules_use_native_normal_and_position(device):
    models, datas = [], []
    for radius in (0.03, 0.05):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep=".005" iterations="100" tolerance="1e-8" gravity="0 0 0"/>
          <worldbody>
            <geom type="capsule" size=".03 .3" pos="0 0 1" zaxis=".3 0 .5"/>
            <body pos="0 0 1"><freejoint/>
              <geom type="capsule" size="{radius} .3" zaxis="-.4 0 .5"/>
            </body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        models.append(model)
        datas.append(data)
    _check_contact_step(models, datas, device)
