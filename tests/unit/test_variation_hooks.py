"""Per-lane physics knobs and the reset rule: unnamed Variation leaves return to their defaults."""

import copy
import importlib
import os
import tempfile

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs._core.physics import GeomMass, ModelField, PhysicsKnobs, state_key

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


def _model_arrays(m):
    out = {}
    for name in dir(m):
        value = None if name.startswith("_") else getattr(m, name, None)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
            out[name] = np.array(value)
    for group in ("opt", "stat"):
        holder = getattr(m, group)
        for name in dir(holder):
            value = None if name.startswith("_") else getattr(holder, name)
            if isinstance(value, (int, float, np.ndarray)) and not isinstance(value, bool):
                out[f"{group}.{name}"] = np.array(value, np.float64)
    return out


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


def test_model_fields_write_through_a_transform():
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
    with pytest.raises(KeyError, match="no geom named"):
        PhysicsKnobs([ModelField("x", "geom_friction", "geom", "missing", 0)], m)


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
        with pytest.raises(ValueError, match="does not record the physics"):
            local.set_state(saved.exclude(state_key("gravity.y")))
        expected, actual = local.step(action), remote.step(action)
        for key in expected.keys(True, True):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0, msg=str(key))
    finally:
        local.close()
        remote.close()


def _gravity(env):
    state = env.get_state()
    sim = np.stack([e.physics.model.opt.gravity for e in env._simulator.envs])
    style = np.concatenate([env._style[f"gravity.{axis}"].reshape(-1, 1).cpu().numpy() for axis in "xyz"], axis=1)
    rows = torch.cat([state["physics", "gravity", axis] for axis in "xyz"], dim=1).numpy()
    np.testing.assert_array_equal(rows, sim)
    np.testing.assert_array_equal(style, sim.astype(np.float32))
    return sim


@pytest.fixture
def cartpole():
    pytest.importorskip("dm_control")
    with se.make("DMCCartpoleSwingup", num_envs=3, device="cpu") as env:
        yield env


def test_reset_sets_or_samples_named_physics_and_restores_the_rest(cartpole):
    assert cartpole.physics_variation_leaves == ("gravity.x", "gravity.y", "gravity.z")
    registered = np.tile([0.0, 0.0, -9.81], (3, 1))
    gz = np.asarray([[-1.1], [-2.2], [-19.99]])
    cartpole.reset(seed=1, options={"variation_values": {"gravity.z": gz}})
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2:], gz)
    cartpole.reset(seed=1, options={"variation_values": {"gravity.x": [1.25]}})
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([1.25, 0.0, -9.81], (3, 1)))
    first = cartpole.reset(seed=9, options={"variation": ["gravity.y"]})["physics", "gravity", "y"]
    again = cartpole.reset(seed=9, options={"variation": ["gravity.y"]})["physics", "gravity", "y"]
    assert torch.equal(first, again) and len(set(first.reshape(-1).tolist())) == 3
    np.testing.assert_array_equal(_gravity(cartpole)[:, [0, 2]], registered[:, [0, 2]])
    cartpole.reset()
    np.testing.assert_array_equal(_gravity(cartpole), registered)
    with pytest.raises(NotImplementedError, match="sampled variation leaves"):
        cartpole.reset(options={"variation": ["agent.color"]})
    with pytest.raises(NotImplementedError, match="variation_values"):
        cartpole.reset(options={"variation_values": {"material.self": [1.0, 0.0, 0.0]}})


def test_set_variation_sets_the_default_and_a_restored_state_does_not(cartpole):
    gz = np.asarray([[-2.0], [-3.0], [-4.0]])
    cartpole.set_variation({"gravity.z": [-6.0]})
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2], [-6.0] * 3)
    saved = cartpole.reset(seed=6, options={"variation_values": {"gravity.z": gz}})
    cartpole.reset(seed=6)
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([0.0, 0.0, -6.0], (3, 1)))
    cartpole.set_state(saved)
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2:], gz)
    cartpole.reset(seed=6)
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([0.0, 0.0, -6.0], (3, 1)))


def test_style_leaves_follow_the_rule():
    with se.make("TwoRoom", num_envs=2, device="cpu") as env:
        registered = float(env._style["agent.speed"][0])
        env.reset(seed=1, options={"variation_values": {"agent.speed": [6.5]}})
        assert (env._style["agent.speed"] == 6.5).all()
        env.reset(seed=1)
        assert (env._style["agent.speed"] == registered).all()

        env.set_variation({"agent.speed": [4.0], "agent.color": [10, 20, 30]})
        env.reset(seed=2, options={"variation_values": {"agent.speed": [[6.5], [7.0]]}})
        assert env._style["agent.speed"].tolist() == [6.5, 7.0]
        assert env._style["agent.color"].tolist() == [[10.0, 20.0, 30.0]] * 2
        env.reset(seed=2)
        assert (env._style["agent.speed"] == 4.0).all()
        assert env._style["agent.color"].tolist() == [[10.0, 20.0, 30.0]] * 2


def test_reset_leaves_take_one_value_per_lane_and_follow_the_rule():
    starts = np.asarray([[31.25, 47.5], [180.0, 60.0]], np.float32)
    with se.make("TwoRoom", num_envs=2, device="cpu") as env:
        assert [leaf.path for leaf in env.reset_leaves] == ["agent.position", "target.position"]
        sampled = env.reset(seed=3)["agent", "position"]
        state = env.reset(seed=3, options={"variation_values": {"agent.position": starts}})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), starts)
        assert torch.equal(env.reset(seed=3)["agent", "position"], sampled)
        state = env.reset(seed=3, options={"variation": []})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), [[60.0, 112.0]] * 2)
        np.testing.assert_array_equal(state["target", "position"].numpy(), [[164.0, 112.0]] * 2)
        with pytest.raises(ValueError, match="per lane"):
            env.reset(options={"variation_values": {"agent.position": np.zeros((3, 2))}})
    with se.make("SimplePointMaze", num_envs=2, device="cpu") as env:
        goals = env.reset(seed=4, options={"variation": ["goal.position"]})["goal", "position"].numpy()
        assert len({tuple(row) for row in goals.tolist()} - {(4.5, 4.5)}) == 2
        state = env.reset(seed=4, options={"variation": [], "variation_values": {"goal.position": [2.5, 1.5]}})
        np.testing.assert_array_equal(state["goal", "position"].numpy(), [[2.5, 1.5]] * 2)


def test_reset_refuses_leaves_it_cannot_set_on_envs_without_reset_options():
    with se.make("CartPole", num_envs=2, device="cpu") as env:
        env.set_variation({"physics.gravity": [12.0]})
        env.reset(seed=0)
        assert (env._style["physics.gravity"] == 12.0).all()
        with pytest.raises(NotImplementedError, match="sampled variation leaves"):
            env.reset(seed=0, options={"variation": ["pole.color"]})


def test_fetch_reset_restores_colors_and_mass():
    pytest.importorskip("gymnasium_robotics")
    options = {
        "variation_values": {
            "table.color": np.array([0.2, 0.7, 0.4]),
            "object.color": np.array([0.8, 0.1, 0.3]),
            "block.mass": np.array([0.9]),
            "background.color": np.array([0.1, 0.2, 0.3]),
            "rendering.transparent_arm": np.array([1, 0]),
            "agent.start_position": np.array([1.4, 0.85]),
        }
    }
    with (
        se.make("FetchPush", num_envs=2, device="cpu", backend="opengl") as env,
        se.make("FetchPush", num_envs=2, device="cpu", backend="opengl") as fresh,
    ):
        registered = env.get_variation().clone()
        mass = env._simulator.object_mass
        env.reset(seed=1, options=options)
        assert not torch.equal(env.get_variation(), registered)
        assert (env._simulator.object_mass == 0.9).all()
        assert env._style["rendering.transparent_arm"].reshape(-1).tolist() == [1.0, 0.0]
        env.reset(seed=1)
        assert torch.equal(env.get_variation(), registered)
        np.testing.assert_array_equal(env._simulator.object_mass, mass)
        assert torch.equal(
            env.get_state()["task", "initial_gripper_xpos"], fresh.get_state()["task", "initial_gripper_xpos"]
        )
        only_object = {"variation_values": {"object.color": np.array([0.1, 0.2, 0.3])}}
        assert torch.equal(
            env.reset(seed=2, options=only_object)["qpos"], fresh.reset(seed=2, options=only_object)["qpos"]
        )
        assert torch.equal(env.get_variation(), fresh.get_variation())
