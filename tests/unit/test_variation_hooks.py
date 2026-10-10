"""Per-lane reset distributions and per-lane physics model parameters."""

import copy
import importlib
import os
import tempfile

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs._core.physics import GeomMass, ModelField, PhysicsKnobs, PhysicsLanes, _model_arrays, state_key

mujoco = pytest.importorskip("mujoco")

#: Arrays a recompile and GeomMass must agree on. body_iquat and bvh_aabb are excluded: they are a
#: principal-axis convention, and a differently signed eigenbasis is the same physical body.
_PHYSICAL = (
    "body_mass",
    "body_ipos",
    "body_subtreemass",
    "body_invweight0",
    "dof_M0",
    "dof_invweight0",
    "actuator_acc0",
    "cam_poscom0",
    "light_poscom0",
    "stat.meaninertia",
    "stat.meanmass",
)


def _compile(domain, geom=None, density=None, keep_total=True):
    pytest.importorskip("dm_control")
    from dm_control import mjcf

    module = importlib.import_module(f"dm_control.suite.{domain}")
    xml, assets = module.get_model_and_assets()
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(module.__file__),
        assets=assets or {},
    )
    if density is not None:
        model.find("geom", geom).density = density
    if not keep_total:
        model.compiler.settotalmass = None
    with tempfile.TemporaryDirectory() as directory:
        mjcf.export_with_assets(model, directory, out_file_name="m.xml")
        return mujoco.MjModel.from_xml_path(os.path.join(directory, "m.xml"))


def _full_inertia(m):
    out = []
    for b in range(m.nbody):
        r = np.empty(9)
        mujoco.mju_quat2Mat(r, m.body_iquat[b])
        r = r.reshape(3, 3)
        out.append(r @ np.diag(m.body_inertia[b]) @ r.T)
    return np.stack(out)


def _roll(m, steps=300):
    d = mujoco.MjData(m)
    d.qvel[:] = 0.1
    rng = np.random.default_rng(0)
    for _ in range(steps):
        d.ctrl[:] = rng.uniform(-1, 1, m.nu)
        mujoco.mj_step(m, d)
    return d.qpos.copy()


@pytest.mark.parametrize("domain", ["cheetah", "hopper", "walker"])
def test_geom_mass_matches_a_recompile(domain):
    """Cheetah compiles with settotalmass; cheetah's and hopper's torso bodies carry two geoms."""
    nominal = _compile(domain, "torso", 1000.0)
    options = {}
    if domain == "cheetah":
        options = {"total_mass": 14.0, "unscaled_total": float(_compile(domain, keep_total=False).body_mass[1:].sum())}
    knobs = PhysicsKnobs([GeomMass("agent.torso_density", "torso", 1000.0)], nominal, **options)
    for density in (500.0, 1500.0):
        recompiled = _compile(domain, "torso", density)
        mine = copy.copy(nominal)
        knobs.apply(mine, knobs.as_row({"agent.torso_density": density}))
        ours, theirs = _model_arrays(mine), _model_arrays(recompiled)
        for name in _PHYSICAL:
            np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-12, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(_full_inertia(mine), _full_inertia(recompiled), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_roll(mine), _roll(recompiled), atol=1e-10)
    restored = copy.copy(nominal)
    knobs.apply(restored, knobs.as_row({"agent.torso_density": 700.0}))
    knobs.apply(restored, knobs.nominal)
    before, after = _model_arrays(nominal), _model_arrays(restored)
    assert all(np.array_equal(before[k], after[k]) for k in before), "the registered row must restore every array"


def test_model_fields_write_through_a_transform_and_report_what_they_touch():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="b")
    joint = body.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE)
    joint.frictionloss = 0.1
    body.add_geom(name="g", size=[0.1, 0, 0])
    m = spec.compile()
    knobs = PhysicsKnobs(
        [
            ModelField("gravity.z", "opt.gravity", index=2),
            ModelField("floor.friction", "geom_friction", "geom", "g", 0),
            ModelField("spinner.friction", "dof_frictionloss", "joint", "hinge", scale=2.0),
        ],
        m,
    )
    np.testing.assert_allclose(knobs.nominal, [-9.81, 1.0, 0.05])
    knobs.apply(m, knobs.as_row({"gravity.z": -3.0, "floor.friction": 0.4, "spinner.friction": 0.25}))
    assert m.opt.gravity[2] == -3.0 and m.geom_friction[0, 0] == 0.4 and m.dof_frictionloss[0] == 0.5
    assert set(knobs.touched_fields()) == {"opt.gravity", "geom_friction", "dof_frictionloss"}
    arrays = knobs.model_arrays(np.asarray([knobs.nominal, [-1.0, 0.7, 0.3]]))
    np.testing.assert_allclose(arrays["opt.gravity"][:, 2], [-9.81, -1.0])
    np.testing.assert_allclose(arrays["dof_frictionloss"][:, 0], [0.1, 0.6])
    with pytest.raises(KeyError, match="no geom named"):
        PhysicsKnobs([ModelField("x", "geom_friction", "geom", "missing", 0)], m)


def test_lanes_carry_physics_in_state_and_refuse_a_state_without_it():
    from stable_envs.envs.dmcontrol.simulation.first_family import FirstFamilyAdapter

    pytest.importorskip("dm_control")
    adapter = FirstFamilyAdapter("cartpole", 2)
    try:
        layout = adapter.physics_layout
        assert layout.leaves == ("gravity.x", "gravity.y", "gravity.z")
        rows = np.asarray([[0.0, 0.0, -9.81], [0.5, -0.25, -4.0]])
        adapter.configure_physics(rows)
        state = adapter.get_state()
        assert state[state_key("gravity.z")].dtype == torch.float64
        np.testing.assert_array_equal(state[state_key("gravity.z")].numpy().reshape(-1), [-9.81, -4.0])
        adapter.configure_physics(np.tile([0.0, 0.0, -9.81], (2, 1)))
        adapter.set_state(state)
        np.testing.assert_array_equal(np.stack([e.physics.model.opt.gravity for e in adapter.envs]), rows)
        with pytest.raises(ValueError, match="does not record the physics"):
            adapter.set_state(state.exclude(state_key("gravity.y")))
    finally:
        adapter.close()


def test_process_shards_carry_and_restore_per_lane_physics():
    from stable_envs.envs.dmcontrol.simulation.process import DMCProcessAdapter, make_local

    pytest.importorskip("dm_control")
    local = make_local("cartpole", 4, seed=5)
    remote = DMCProcessAdapter("cartpole", 4, seed=5, processes=2)
    try:
        assert remote.physics_layout == local.physics_layout
        rows = np.asarray([[0.0, 0.0, -9.81], [0.0, 0.0, -2.0], [1.0, 0.0, -9.81], [0.0, 0.0, -15.0]])
        for owner in (local, remote):
            owner.configure_physics(rows)
        action = np.full((4, 1), 0.4)
        expected, actual = local.step(action), remote.step(action)
        for key in expected.keys(True, True):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0, msg=str(key))
        saved = remote.get_state()
        for owner in (local, remote):
            owner.configure_physics(np.tile([0.0, 0.0, -9.81], (4, 1)))
            owner.set_state(saved)
        torch.testing.assert_close(remote.get_state()[state_key("gravity.z")], saved[state_key("gravity.z")])
        expected, actual = local.step(action), remote.step(action)
        for key in expected.keys(True, True):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0, msg=str(key))
    finally:
        local.close()
        remote.close()


def test_reset_options_set_or_sample_physics_per_lane():
    pytest.importorskip("dm_control")
    with se.make("DMCCartpoleSwingup", num_envs=3, device="cpu") as env:
        assert env.physics_variation_leaves == ("gravity.x", "gravity.y", "gravity.z")
        exact = np.asarray([[-1.1], [-2.2], [-19.99]])
        state = env.reset(seed=1, options={"variation_values": {"gravity.z": exact}})
        np.testing.assert_array_equal(state["physics", "gravity", "z"].numpy(), exact)
        np.testing.assert_array_equal([e.physics.model.opt.gravity[2] for e in env._simulator.envs], exact[:, 0])
        first = env.reset(seed=9, options={"variation": ["gravity.z"]})["physics", "gravity", "z"]
        again = env.reset(seed=9, options={"variation": ["gravity.z"]})["physics", "gravity", "z"]
        assert torch.equal(first, again) and len(set(first.reshape(-1).tolist())) == 3
        assert float(env.reset(seed=2)["physics", "gravity", "z"][0]) == float(first[0])
        with pytest.raises(NotImplementedError, match="sampled variation leaves"):
            env.reset(options={"variation": ["agent.color"]})
        with pytest.raises(NotImplementedError, match="variation_values"):
            env.reset(options={"variation_values": {"material.self": [1.0, 0.0, 0.0]}})


def test_reset_leaves_take_one_value_per_lane():
    starts = np.asarray([[31.25, 47.5], [180.0, 60.0], [40.0, 200.0]], np.float32)
    with se.make("TwoRoom", num_envs=3, device="cpu") as env:
        assert [leaf.path for leaf in env.reset_leaves] == ["agent.position", "target.position"]
        state = env.reset(seed=3, options={"variation_values": {"agent.position": starts}})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), starts)
        with pytest.raises(ValueError, match="per lane"):
            env.reset(options={"variation_values": {"agent.position": np.zeros((2, 2))}})
    with se.make("SimplePointMaze", num_envs=2, device="cpu") as env:
        with pytest.raises(NotImplementedError, match="goal.position"):
            env.reset(options={"variation": ["goal.position"]})
        state = env.reset(seed=4, options={"variation": [], "variation_values": {"goal.position": [2.5, 1.5]}})
        np.testing.assert_array_equal(state["goal", "position"].numpy(), [[2.5, 1.5]] * 2)


def test_lanes_rebind_reapplies_varied_values():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body().add_geom(size=[0.1, 0, 0])
    models = [spec.compile(), spec.compile()]
    lanes = PhysicsLanes([ModelField("gravity.z", "opt.gravity", index=2)], models)
    lanes.configure({"gravity.z": np.asarray([[-9.81], [-2.0]])})
    fresh = [spec.compile(), spec.compile()]
    lanes.rebind(fresh)
    assert fresh[1].opt.gravity[2] == -2.0 and fresh[0].opt.gravity[2] == -9.81
