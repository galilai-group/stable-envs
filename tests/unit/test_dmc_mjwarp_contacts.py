"""Core regressions for MuJoCo Warp 3.17 contact geometry and constraint ordering."""

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
    with closing(WarpRuntime(models, device, 1)) as runtime:
        runtime.load(state.to(device))
        count = int(runtime._d.nacon.numpy()[0])
        contacts = runtime._d.contact
        worlds = contacts.worldid.numpy()[:count]
        active = contacts.type.numpy()[:count] != 0
        # Touch sensors use constraint addresses, including for inactive slots.
        assert (contacts.efc_address.numpy()[:count][~active] == -1).all()
        np.testing.assert_array_equal(runtime._d.nefc.numpy(), [data.nefc for data in datas])
        for world, data in enumerate(datas):
            indices = np.flatnonzero(active & (worlds == world))
            indices = indices[np.argsort(contacts.geomcollisionid.numpy()[indices])]
            assert len(indices) == data.ncon
            for index, expected in zip(indices, data.contact, strict=True):
                for name in ("frame", "pos", "dist"):
                    actual = getattr(contacts, name).numpy()[index]
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
        np.testing.assert_allclose(
            runtime._d.sensordata.numpy(), np.stack([data.sensordata for data in datas]), atol=4e-5, rtol=4e-5
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


@pytest.mark.parametrize("device", DEVICES)
def test_near_parallel_capsules_preserve_contacts_and_impulses(device):
    models, datas = [], []
    # MuJoCo 3.17 treats these small tilts as parallel and keeps both contacts;
    # the separated lane must receive neither impulses nor touch.
    cases = [(tilt, offset, [0, 0, 1]) for tilt, offset in ((3e-5, 0.055), (-3e-5, 0.055), (0, 0.055), (3e-5, 1))]
    # Rotations can make upstream's cancelling float32 determinant emit only
    # one contact. Both sides of the new native angular cutoff are covered.
    cases += [(tilt, 0.055, [0.6, 0.8, 1]) for tilt in (3e-5, 8e-5, 1.2e-4, -1.2e-4)]
    # Only one endpoint lies inside the contact detection band in these lanes.
    cases += [(3e-5, offset, [0, 0, 1]) for offset in (0.059998, 0.060002)]
    for tilt, offset, axis in cases:
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        side = np.cross([0, 1, 0], axis)
        side /= np.linalg.norm(side)
        fmt = lambda values: " ".join(map(str, values))
        model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep=".005" iterations="100" tolerance="1e-8" gravity="0 0 0"/>
          <worldbody><geom type="capsule" size=".03 .125" zaxis="{fmt(axis)}"/>
            <body pos="{fmt(offset * side + 0.05 * axis)}"><freejoint/>
              <geom type="capsule" size=".03 .125" zaxis="{fmt(axis + tilt * side)}"/>
              <site name="touch" type="sphere" size=".3"/>
            </body>
          </worldbody><sensor><touch site="touch"/></sensor></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = [0.1, 0.2, -0.1, 0.05, 0.1, 0]
        mujoco.mj_forward(model, data)
        models.append(model)
        datas.append(data)
    assert [data.ncon for data in datas] == [2, 2, 2, 0, 2, 2, 1, 1, 1, 1]
    _check_contact_step(models, datas, device)

    # A gap keeps a detected endpoint without adding its constraint. Explicit
    # pair gaps override geom gaps; both routes must match native detection.
    for explicit in (False, True):
        models, datas = [], []
        for gap in (0.0, 1e-5):
            contact = f'<contact><pair geom1="a" geom2="b" margin="2e-6" gap="{gap}"/></contact>' if explicit else ""
            geom_gap = 9e-4 if explicit else gap / 2
            model = mujoco.MjModel.from_xml_string(f'''<mujoco>
              <option gravity="0 0 0"/>
              <default><geom type="capsule" size=".03 .125" margin="1e-6" gap="{geom_gap}"/></default>
              <worldbody><geom name="a"/><body pos=".060005 0 .05"><freejoint/>
                <geom name="b" zaxis=".00003 0 1"/>
              </body></worldbody>{contact}</mujoco>''')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            models.append(model)
            datas.append(data)
        assert [data.ncon for data in datas] == [1, 2]
        _check_contact_step(models, datas, device)


@pytest.mark.parametrize("device", DEVICES)
def test_constraint_order_preserves_contact_impulses(device):
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".005" cone="pyramidal" iterations="100" tolerance="1e-8"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .045"><freejoint/>
          <geom type="capsule" fromto="-.2 0 0 .2 0 0" size=".05"/>
        </body>
        <body pos="0 0 1"><joint limited="true" range="-20 20"/>
          <geom type="capsule" fromto="0 0 0 .2 0 0" size=".03"/>
        </body>
      </worldbody></mujoco>""")
    datas = [mujoco.MjData(model) for _ in range(2)]
    for lane, data in enumerate(datas):
        data.qpos[-1] = 0.5
        data.qpos[2] += lane * 0.5
        data.qvel[:] = np.linspace(-0.3, 0.2, model.nv)
        mujoco.mj_forward(model, data)
    assert [data.ncon for data in datas] == [2, 0]
    state = TensorDict(
        {
            "qpos": torch.tensor(np.stack([d.qpos for d in datas])),
            "qvel": torch.tensor(np.stack([d.qvel for d in datas])),
            "time": torch.zeros(2, 1),
            "warmstart": torch.zeros(2, model.nv),
        },
        [2],
    ).to(device)
    with closing(WarpRuntime([model, model], device, 1, ordered_constraints=True)) as runtime:
        import warp as wp

        runtime.load(state)
        controls = torch.empty(2, 0, device=device)
        runtime.advance(controls)
        expected = runtime.physical_state()
        runtime.load(state)
        with runtime._scope():
            data = runtime._d
            count = int(data.nefc.numpy()[0])
            assert count == 9
            rows = runtime._constraint_buffers[0]
            before = {name: wp.to_torch(getattr(rows, name)).clone() for name in rows._cls.vars}
            addresses = wp.to_torch(data.contact.efc_address)
            old_addresses = addresses.clone()
            # Exchange lane 0's contact blocks and move its joint limit last,
            # retaining the four local friction rows. Lane 1 stays intact.
            order = torch.tensor([5, 6, 7, 8, 1, 2, 3, 4, 0], device=device)
            inverse = torch.argsort(order)
            for name, saved in before.items():
                wp.to_torch(getattr(rows, name))[0, :count].copy_(saved[0, order])
            world = wp.to_torch(data.contact.worldid)
            active = torch.arange(data.naconmax, device=device) < wp.to_torch(data.nacon)[0]
            mask = active[:, None] & (world[:, None] == 0) & (old_addresses >= 0)
            addresses.copy_(torch.where(mask, inverse[old_addresses.clamp(0, count - 1).long()], old_addresses))
            runtime._order_constraints(runtime._m, data)
            for name, saved in before.items():
                torch.testing.assert_close(wp.to_torch(getattr(rows, name)), saved, atol=0, rtol=0, equal_nan=True)
            torch.testing.assert_close(addresses, old_addresses, atol=0, rtol=0)
        runtime.advance(controls)
        for name, value in expected.items():
            torch.testing.assert_close(runtime.physical_state()[name], value, atol=0, rtol=0)
        for data in datas:
            mujoco.mj_step2(model, data)
        np.testing.assert_allclose(expected["qvel"].cpu(), np.stack([d.qvel for d in datas]), atol=4e-5, rtol=4e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_elliptic_contact_reduction_preserves_impulses(device):
    import warp as wp

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".005" cone="elliptic" solver="Newton" iterations="100" tolerance="1e-8"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .045"><freejoint/>
          <geom type="sphere" size=".05" pos="-.15 -.1 0"/>
          <geom type="sphere" size=".05" pos=".15 -.1 0"/>
          <geom type="sphere" size=".05" pos="0 .15 0"/>
        </body>
      </worldbody></mujoco>""")
    datas = [mujoco.MjData(model) for _ in range(2)]
    for lane, data in enumerate(datas):
        data.qpos[2] += lane
        data.qvel[:] = [1, 0.7, -0.3, 0.1, 0.2, 0.3]
        mujoco.mj_forward(model, data)
    state = TensorDict(
        {
            "qpos": torch.tensor(np.stack([d.qpos for d in datas])),
            "qvel": torch.tensor(np.stack([d.qvel for d in datas])),
            "time": torch.zeros(2, 1),
            "warmstart": torch.zeros(2, model.nv),
        },
        [2],
    ).to(device)
    with closing(WarpRuntime([model, model], device, 1, ordered_elliptic=True)) as runtime:
        runtime.load(state)
        with runtime._scope():
            data, solver = runtime._d, runtime._elliptic_solver
            assert data.nefc.numpy().tolist() == [9, 0]
            assert (data.efc.state.numpy()[0, :9] == mujoco.mjtConstraintState.mjCNSTRSTATE_CONE).all()
            solver.ctx.done.zero_()
            solver.gradient()
            expected_h = wp.to_torch(solver.ctx.h).clone()
            # Contact storage may be allocated in a different order on CUDA.
            # Keep physical constraint rows fixed and reverse their contact IDs.
            order = torch.tensor([2, 1, 0], device=device)
            for name in ("friction", "dim", "efc_address"):
                view = wp.to_torch(getattr(data.contact, name))
                view[:3].copy_(view[:3][order])
            ids = wp.to_torch(data.efc.id)
            ids[0, :9] = order[ids[0, :9].long()]
            for _ in range(10):
                solver.gradient()
                torch.testing.assert_close(wp.to_torch(solver.ctx.h), expected_h, atol=0, rtol=0)
        controls = torch.empty(2, 0, device=device)
        runtime.load(state)
        runtime.advance(controls)
        expected = runtime.physical_state()
        runtime.load(state)
        runtime.advance(controls)
        for name, value in expected.items():
            torch.testing.assert_close(runtime.physical_state()[name], value, atol=0, rtol=0)
        for data in datas:
            mujoco.mj_step2(model, data)
        np.testing.assert_allclose(expected["qvel"].cpu(), np.stack([d.qvel for d in datas]), atol=4e-5, rtol=4e-5)
