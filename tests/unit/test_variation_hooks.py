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


def _compile(domain, geom=None, density=None, keep_total=True, size=None, tilt=None):
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
    if size is not None:
        model.find("geom", geom).size = [size]
    if not keep_total:
        model.compiler.settotalmass = None
    if tilt is not None:
        angle = np.deg2rad(tilt)
        model.find("geom", "floor").zaxis = np.array([np.sin(angle), 0.0, np.cos(angle)], np.float32)
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


def test_sphere_radius_and_density_match_a_recompile():
    from stable_envs.envs.dmcontrol.simulation.physics import SphereKnobs, knobs

    nominal = _compile("ball_in_cup")
    recompiled = _compile("ball_in_cup", "ball", 1300.0, size=0.04)
    spec = SphereKnobs(knobs("ball_in_cup"), nominal)
    mine = copy.copy(nominal)
    spec.apply(mine, spec.as_row({"ball.density": 1300.0, "ball.size": 0.04}))
    ours, theirs = _model_arrays(mine), _model_arrays(recompiled)
    for name in (*_PHYSICAL, "body_inertia", "geom_size", "geom_rbound", "geom_aabb"):
        np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-12, atol=1e-14, err_msg=name)
    np.testing.assert_allclose(_roll(mine), _roll(recompiled), atol=1e-10)
    spec.apply(mine, spec.nominal)
    before, after = _model_arrays(nominal), _model_arrays(mine)
    assert all(np.array_equal(before[k], after[k]) for k in before), "the registered row must restore every array"


def test_floor_tilt_matches_a_recompile():
    from stable_envs.envs.dmcontrol.simulation.physics import TiltKnobs, knobs

    nominal = _compile("walker")
    spec = TiltKnobs(knobs("walker"), nominal)
    for degrees in (-7.3, 1e-9, 9.0):
        mine, recompiled = copy.copy(nominal), _compile("walker", tilt=degrees)
        spec.apply(mine, spec.as_row({"floor.rotation_y": degrees}))
        ours, theirs = _model_arrays(mine), _model_arrays(recompiled)
        for name in ours:
            np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-14, atol=0, err_msg=name)
        np.testing.assert_array_equal(mine.geom_quat, recompiled.geom_quat)
        np.testing.assert_array_equal(mine.geom_sameframe, recompiled.geom_sameframe)
    spec.apply(mine, spec.nominal)
    before, after = _model_arrays(nominal), _model_arrays(mine)
    assert all(np.array_equal(before[k], after[k]) for k in before), "the registered row must restore every array"


def test_geom_mass_scales_a_body_it_covers_without_rotating_its_frame():
    """Every ball-in-cup cup part shares one leaf, so the cup's inertial frame survives."""
    from stable_envs.envs.dmcontrol.simulation.physics import knobs

    nominal = _compile("ball_in_cup")
    spec = PhysicsKnobs([k for k in knobs("ball_in_cup") if isinstance(k, GeomMass)], nominal)
    mine = copy.copy(nominal)
    spec.apply(mine, spec.as_row({"agent.density": 700.0}))
    body = int(nominal.geom_bodyid[mujoco.mj_name2id(nominal, mujoco.mjtObj.mjOBJ_GEOM, "cup_part_0")])
    np.testing.assert_allclose(mine.body_mass[body], nominal.body_mass[body] * 0.7, rtol=1e-14)
    np.testing.assert_array_equal(mine.body_iquat[body], nominal.body_iquat[body])
    np.testing.assert_array_equal(mine.body_ipos[body], nominal.body_ipos[body])


@pytest.mark.parametrize("cell", ["DMCCheetahRun", "DMCQuadrupedWalk"])
def test_render_masses_match_the_simulated_model(cell):
    """The tracking camera's per-row masses are the ones PhysicsKnobs writes into the simulator."""
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol import _mass_rows
    from stable_envs.envs.dmcontrol.simulation.physics import compile_options, knobs

    cls = se.registry.REGISTRY[cell]
    key, model = cls._simulator_key(), cls._model()
    rows = _mass_rows.MassRows(key, model)
    spec = PhysicsKnobs(knobs(key), model, **compile_options(key))
    values = {rows.paths[0]: 650.0, rows.paths[1]: 1400.0}
    mine = copy.copy(model)
    spec.apply(mine, spec.as_row(values))
    got = rows(torch.tensor([[values[p] for p in rows.paths]]))
    for name in _mass_rows.ARRAYS:
        np.testing.assert_allclose(got[name][0].numpy(), getattr(mine, name), rtol=1e-12, atol=1e-12, err_msg=name)
    assert not np.allclose(mine.cam_poscom0, model.cam_poscom0), "the leaves must move the tracking camera"


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


def test_dmc_batch_rejects_a_state_without_its_physics():
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol.simulation.factory import make_batch

    adapter = make_batch("cartpole", 2)
    try:
        stale = adapter.get_state().exclude(state_key("gravity.y"))
        current = adapter.step(np.full((2, 1), 0.4))["state"]
        with pytest.raises(ValueError, match="does not record the physics"):
            adapter.set_state(stale)
        assert (adapter.get_state() == current).all()
    finally:
        adapter.close()


def _gravity(env):
    state = env.get_state()
    sim = np.stack([e.physics.model.opt.gravity for e in env._simulator.lanes.envs])
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
    assert cartpole.physics_variation_leaves == ("gravity.x", "gravity.y", "gravity.z", "agent.cart_mass")
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


def test_a_state_draws_its_own_frame_moving_physics():
    """Ball-in-cup draws the ball at its radius: another env renders the State as its producer does."""
    pytest.importorskip("dm_control")
    sizes = np.asarray([[0.012], [0.048]])
    with (
        se.make("DMCBallInCupCatch", num_envs=2, device="cpu") as env,
        se.make("DMCBallInCupCatch", num_envs=2, device="cpu") as other,
    ):
        state = env.reset(seed=3, options={"variation_values": {"ball.size": sizes}})
        frames = env.render(state)
        assert torch.equal(other.render(state), frames)
        assert not torch.equal(other.render(state, variation={"ball.size": [0.025]}), frames)


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


def test_reset_refuses_leaves_it_cannot_set_on_envs_without_reset_options():
    with se.make("CartPole", num_envs=2, device="cpu") as env:
        env.set_variation({"physics.gravity": [12.0]})
        env.reset(seed=0)
        assert (env._style["physics.gravity"] == 12.0).all()
        with pytest.raises(NotImplementedError, match="sampled variation leaves"):
            env.reset(seed=0, options={"variation": ["pole.color"]})


@pytest.mark.parametrize(
    "name,leaf,default,values",
    [
        ("CartPole", "physics.gravity", 12.0, [5.5, 14.0]),
        ("Pendulum", "physics.g", 12.0, [5.5, 14.0]),
        ("Acrobot", "physics.link_mass_2", 1.2, [0.6, 1.4]),
        ("MountainCar", "physics.force", 0.0012, [0.0006, 0.0014]),
        ("MountainCarContinuous", "physics.power", 0.0018, [0.0008, 0.0022]),
    ],
)
def test_classic_state_restores_its_physics_for_the_episode(name, leaf, default, values):
    key = state_key(leaf.removeprefix("physics."))
    with se.make(name, num_envs=2, device="cpu") as env:
        action = torch.full((2, 1), env.action_spec.high, dtype=env.action_spec.dtype)
        env.set_variation({leaf: [default]})
        saved = env.reset(seed=5, options={"variation_values": {leaf: np.asarray([[v] for v in values], np.float32)}})
        assert saved[key].dtype == torch.float64
        np.testing.assert_array_equal(saved[key].numpy(), np.float32([[v] for v in values]).astype(np.float64))
        rollout = [env.step(action, frames=None)["state"] for _ in range(8)]
        plain = env.reset(seed=5)
        np.testing.assert_array_equal(plain[key].numpy(), np.float32([[default]] * 2).astype(np.float64))
        env.set_state(saved)
        assert env._style[leaf].reshape(-1).tolist() == np.float32(values).tolist()
        for want in rollout:
            assert (env.step(action, frames=None)["state"] == want).all()
        assert torch.equal(env.reset(seed=5)[key], plain[key])
        with pytest.raises(ValueError, match="physics it was produced under"):
            env.set_state(saved.exclude(key))


@pytest.mark.family("fetch")
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
