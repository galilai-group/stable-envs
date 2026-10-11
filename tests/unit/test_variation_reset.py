"""Every reset returns the Variation leaves its options do not name to their defaults.

The defaults are the registered values until ``set_variation`` replaces them.
"""

import numpy as np
import pytest
import torch

import stable_envs as se


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


def test_plain_reset_restores_registered_physics(cartpole):
    registered = np.tile([0.0, 0.0, -9.81], (3, 1))
    gz = np.asarray([[-1.5], [-4.0], [-12.0]])
    cartpole.reset(seed=1, options={"variation_values": {"gravity.z": gz}})
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2:], gz)
    cartpole.reset(seed=1)
    np.testing.assert_array_equal(_gravity(cartpole), registered)
    cartpole.reset(seed=2, options={"variation": ["gravity.x", "gravity.z"]})
    assert len(set(_gravity(cartpole)[:, 0].tolist())) == 3
    cartpole.reset()
    np.testing.assert_array_equal(_gravity(cartpole), registered)


def test_plain_reset_restores_set_variation_physics(cartpole):
    cartpole.set_variation({"gravity.z": [-5.0]})
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2], [-5.0] * 3)
    cartpole.reset(seed=3, options={"variation": ["gravity.z"]})
    assert (_gravity(cartpole)[:, 2] != -5.0).all()
    cartpole.reset(seed=3)
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([0.0, 0.0, -5.0], (3, 1)))


def test_unnamed_leaves_revert_while_named_ones_are_set(cartpole):
    gz = np.asarray([[-2.0], [-3.0], [-4.0]])
    cartpole.reset(seed=4, options={"variation_values": {"gravity.z": gz}})
    cartpole.reset(seed=4, options={"variation_values": {"gravity.x": [1.25]}})
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([1.25, 0.0, -9.81], (3, 1)))
    cartpole.reset(seed=5, options={"variation": ["gravity.y"]})
    gravity = _gravity(cartpole)
    np.testing.assert_array_equal(gravity[:, [0, 2]], np.tile([0.0, -9.81], (3, 1)))
    assert len(set(gravity[:, 1].tolist())) == 3


def test_restored_state_physics_is_not_a_default(cartpole):
    gz = np.asarray([[-2.0], [-3.0], [-4.0]])
    cartpole.set_variation({"gravity.z": [-6.0]})
    saved = cartpole.reset(seed=6, options={"variation_values": {"gravity.z": gz}})
    cartpole.reset(seed=6)
    cartpole.set_state(saved)
    np.testing.assert_array_equal(_gravity(cartpole)[:, 2:], gz)
    cartpole.reset(seed=6)
    np.testing.assert_array_equal(_gravity(cartpole), np.tile([0.0, 0.0, -6.0], (3, 1)))


def test_two_room_reset_restores_registered_and_configured_defaults():
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


def test_reset_leaves_follow_the_rule():
    starts = np.asarray([[31.25, 47.5], [180.0, 60.0]], np.float32)
    with se.make("TwoRoom", num_envs=2, device="cpu") as env:
        sampled = env.reset(seed=3)["agent", "position"]
        env.reset(seed=3, options={"variation_values": {"agent.position": starts}})
        assert torch.equal(env.reset(seed=3)["agent", "position"], sampled)
        state = env.reset(seed=3, options={"variation": []})
        np.testing.assert_array_equal(state["agent", "position"].numpy(), [[60.0, 112.0]] * 2)
        np.testing.assert_array_equal(state["target", "position"].numpy(), [[164.0, 112.0]] * 2)


def test_reset_refuses_leaves_it_cannot_set_on_envs_without_reset_options():
    with se.make("CartPole", num_envs=2, device="cpu") as env:
        env.set_variation({"physics.gravity": [12.0]})
        env.reset(seed=0)
        assert (env._style["physics.gravity"] == 12.0).all()
        with pytest.raises(NotImplementedError, match="sampled variation leaves"):
            env.reset(seed=0, options={"variation": ["physics.gravity"]})


def test_fetch_reset_restores_colors_and_mass():
    pytest.importorskip("gymnasium_robotics")
    options = {
        "variation_values": {
            "table.color": np.array([0.2, 0.7, 0.4]),
            "object.color": np.array([0.8, 0.1, 0.3]),
            "block.mass": np.array([0.9]),
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
        env.reset(seed=1)
        assert torch.equal(env.get_variation(), registered)
        np.testing.assert_array_equal(env._simulator.object_mass, mass)
        only_object = {"variation_values": {"object.color": np.array([0.1, 0.2, 0.3])}}
        assert torch.equal(
            env.reset(seed=2, options=only_object)["qpos"], fresh.reset(seed=2, options=only_object)["qpos"]
        )
        assert torch.equal(env.get_variation(), fresh.get_variation())
