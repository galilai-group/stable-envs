"""Native primitive witnesses, independent of task-level calibrated bounds."""

from contextlib import closing

import numpy as np
import pytest
import torch
from tensordict import TensorDict

mujoco = pytest.importorskip("mujoco")

from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import WarpRuntime  # noqa: E402

pytestmark = pytest.mark.family("dmc")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    pytest.importorskip("mujoco_warp")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA contact kernels")
    return request.param


def source(normal, axis, kind="capsule", extra=""):
    import mujoco

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    center = normal * (0.03 + (0.1 * abs(normal @ axis) if kind == "capsule" else 0) - 0.005)
    fmt = lambda values: " ".join(map(str, values))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".005" cone="pyramidal" iterations="100" tolerance="1e-8"/>
      <worldbody><geom type="plane" zaxis="{fmt(normal)}" size="1 1 .1"/>
        <body pos="{fmt(center)}"><freejoint/>
          <geom type="{kind}" zaxis="{fmt(axis)}" size=".03 .1"/>
        </body>
      {extra}</worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = [1, 0.7, -0.3, 0.1, 0.2, 0.3]
    mujoco.mj_forward(model, data)
    state = TensorDict(
        {
            "qpos": torch.tensor(data.qpos[None].copy()),
            "qvel": torch.tensor(data.qvel[None].copy()),
            "time": torch.zeros(1, 1, dtype=torch.float64),
            "warmstart": torch.tensor(data.qacc_warmstart[None].copy()),
        },
        [1],
    )
    return model, data, state


def frames(runtime):
    count = int(runtime._d.nacon.numpy()[0])
    return runtime._d.contact.frame.numpy()[:count].copy()


@pytest.mark.parametrize(
    "normal,axis",
    [
        ([0, 0, 1], [0.2, 0.2, 1]),
        ([0.2, -0.1, 1], [0.4, 0.1, 1]),
        ([0, 0, 1], [1, 1, 1]),
        ([0, 0, 1], [0, 0, 1]),
        ([0, 0, 1], [1, 0, 0]),
    ],
)
def test_plane_capsule_frames_and_one_step_match_native(normal, axis, device):
    import mujoco

    model, data, state = source(normal, axis)
    expected_frames = np.stack([c.frame.reshape(3, 3) for c in data.contact])
    with closing(WarpRuntime([model], device, 1)) as runtime:
        runtime.load(state.to(device))
        np.testing.assert_allclose(frames(runtime), expected_frames, atol=8e-7, rtol=8e-7)
        # Constraint integration proves the contact frame has a causal effect,
        # not just a cosmetic difference in a diagnostic array.
        mujoco.mj_step2(model, data)
        mujoco.mj_step1(model, data)
        runtime.advance(torch.empty(1, 0, device=device))
        np.testing.assert_allclose(runtime.physical_state()["qpos"][0].cpu(), data.qpos, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(runtime.physical_state()["qvel"][0].cpu(), data.qvel, atol=4e-5, rtol=4e-5)


def test_upstream_contact_frame_fault_is_detected_and_model_callback_is_local(device):
    model, data, state = source([0, 0, 1], [0.2, 0.2, 1])
    expected = np.stack([c.frame.reshape(3, 3) for c in data.contact])
    with closing(WarpRuntime([model], device, 1)) as fixed, closing(WarpRuntime([model], device, 1)) as faulty:
        faulty._m.callback.contactfilter = None
        fixed.load(state.to(device))
        faulty.load(state.to(device))
        np.testing.assert_allclose(frames(fixed), expected, atol=8e-7, rtol=8e-7)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(frames(faulty), expected, atol=8e-7, rtol=8e-7)
        assert np.max(abs(frames(faulty) - expected)) > 0.5


def test_other_primitives_are_unchanged(device):
    model, data, state = source(
        [0.2, -0.1, 1],
        [1, 0, 0],
        kind="sphere",
        extra='<geom type="capsule" size=".01 .05" pos="0 0 2" contype="0" conaffinity="0"/>',
    )
    with closing(WarpRuntime([model], device, 1)) as runtime:
        assert runtime._m.callback.contactfilter is not None
        runtime.load(state.to(device))
        before = frames(runtime)
        with runtime._scope():
            runtime._correct_contact_frames(runtime._m, runtime._d)
        np.testing.assert_array_equal(frames(runtime), before)


@pytest.mark.parametrize("radius2", [0.03, 0.05])
def test_intersecting_capsules_use_native_cross_axis_normal_and_position(radius2, device):
    import mujoco

    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".005" iterations="100" tolerance="1e-8" gravity="0 0 0"/>
      <worldbody>
        <geom type="capsule" size=".03 .3" pos="0 0 1" zaxis=".3 0 .5"/>
        <body pos="0 0 1"><freejoint/>
          <geom type="capsule" size="{radius2} .3" zaxis="-.4 0 .5"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    expected = data.contact[0]
    state = TensorDict(
        {
            "qpos": torch.tensor(data.qpos[None].copy()),
            "qvel": torch.tensor(data.qvel[None].copy()),
            "time": torch.zeros(1, 1, dtype=torch.float64),
            "warmstart": torch.tensor(data.qacc_warmstart[None].copy()),
        },
        [1],
    )
    with closing(WarpRuntime([model], device, 1)) as fixed, closing(WarpRuntime([model], device, 1)) as faulty:
        faulty._m.callback.contactfilter = None
        fixed.load(state.to(device))
        faulty.load(state.to(device))
        np.testing.assert_allclose(frames(fixed)[0], expected.frame.reshape(3, 3), atol=8e-7, rtol=8e-7)
        np.testing.assert_allclose(fixed._d.contact.pos.numpy()[0], expected.pos, atol=8e-7, rtol=8e-7)
        np.testing.assert_allclose(fixed._d.contact.dist.numpy()[0], expected.dist, atol=8e-7, rtol=8e-7)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(frames(faulty)[0], expected.frame.reshape(3, 3), atol=8e-7, rtol=8e-7)
        mujoco.mj_step2(model, data)
        mujoco.mj_step1(model, data)
        fixed.advance(torch.empty(1, 0, device=device))
        np.testing.assert_allclose(fixed.physical_state()["qpos"][0].cpu(), data.qpos, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(fixed.physical_state()["qvel"][0].cpu(), data.qvel, atol=4e-5, rtol=4e-5)


def test_per_world_plane_orientation_is_used(device):
    models = []
    states = []
    expected = []
    for normal, axis in [([0, 0, 1], [0.2, 0.2, 1]), ([0.2, -0.1, 1], [0.4, 0.1, 1])]:
        model, data, state = source(normal, axis)
        models.append(model)
        states.append(state)
        expected.append(data.contact[0].frame.reshape(3, 3).copy())
    with closing(WarpRuntime(models, device, 1)) as runtime:
        runtime.load(torch.cat(states, 0).to(device))
        count = int(runtime._d.nacon.numpy()[0])
        assert count == 2
        worlds = runtime._d.contact.worldid.numpy()[:count].copy()
        for world, actual in zip(worlds, frames(runtime), strict=True):
            np.testing.assert_allclose(actual, expected[world], atol=8e-7, rtol=8e-7)


def test_nonintersecting_capsules_remain_unchanged(device):
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="capsule" size=".03 .3" pos=".1 0 .9" zaxis=".3 0 .5"/>
      <body pos="-.1 .04 1.05"><freejoint/>
        <geom type="capsule" size=".03 .3" zaxis="-.4 0 .5"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = TensorDict(
        {
            "qpos": torch.tensor(data.qpos[None].copy()),
            "qvel": torch.tensor(data.qvel[None].copy()),
            "time": torch.zeros(1, 1, dtype=torch.float64),
            "warmstart": torch.tensor(data.qacc_warmstart[None].copy()),
        },
        [1],
    )
    with closing(WarpRuntime([model], device, 1)) as fixed, closing(WarpRuntime([model], device, 1)) as original:
        original._m.callback.contactfilter = None
        fixed.load(state.to(device))
        original.load(state.to(device))
        assert int(fixed._d.nacon.numpy()[0]) == 1
        for field in ("frame", "dist", "pos"):
            np.testing.assert_array_equal(
                getattr(fixed._d.contact, field).numpy()[0], getattr(original._d.contact, field).numpy()[0]
            )
