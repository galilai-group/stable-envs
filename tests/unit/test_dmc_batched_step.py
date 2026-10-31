"""Batched DMC lanes equal dm_control stepped lane by lane, across implicit resets."""

import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_local  # noqa: E402
from stable_envs.envs.dmcontrol.simulation.state import MODEL_FIELDS  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

LANES = 5
STEPS = 12


def _source_info(key, physics):
    """The task outputs, read through dm_control's Physics methods."""
    if key in ("acrobot", "pendulum", "cartpole", "cartpole-balance", "reacher"):
        return None
    named = physics.named.data
    info = {"qpos": physics.data.qpos.copy(), "qvel": physics.data.qvel.copy()}
    success = float(physics.in_target()) if key == "ball_in_cup" else float("nan")
    if key.startswith("finger"):
        info.update(
            hinge_velocity=np.asarray(physics.hinge_velocity()),
            target_position=physics.target_position().copy(),
            tip_position=physics.tip_position().copy(),
        )
    elif key == "quadruped":
        info["torso_velocity"] = physics.torso_velocity().copy()
    elif key == "cheetah":
        info.update(
            speed=physics.speed(),
            torso_height=named.xpos["torso", "z"],
            bfoot_height=named.xpos["bfoot", "z"],
            ffoot_height=named.xpos["ffoot", "z"],
            angmomentum=named.subtree_angmom["torso"][1],
        )
    elif key == "hopper":
        info.update(speed=physics.speed(), height=physics.height(), angmomentum=named.subtree_angmom["torso"][1])
    elif key == "walker":
        info.update(
            speed=physics.horizontal_velocity(),
            torso_height=physics.torso_height(),
            torso_upright=physics.torso_upright(),
            **{
                f"{name}_height": named.xpos[name, "z"]
                for name in ("left_foot", "right_foot", "left_thigh", "right_thigh")
            },
        )
    elif key == "humanoid":
        info.update(
            speed=np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]]),
            torso_upright=physics.torso_upright(),
            head_height=physics.head_height(),
        )
    return success, {name: np.asarray(value, np.float64) for name, value in info.items()}


def _source_state(lanes, cumulative):
    """State columns read from each lane's dm_control objects."""
    rows = []
    for i, env in enumerate(lanes.envs):
        physics, base = env.physics, env._env
        _, keys, position, has_gauss, cached = env.task.random.get_state()
        row = {
            "qpos": physics.data.qpos.copy(),
            "qvel": physics.data.qvel.copy(),
            "time": np.array([physics.data.time]),
            "warmstart": physics.data.qacc_warmstart.copy(),
            ("lifecycle", "step_count"): np.array([base._step_count], np.int64),
            ("lifecycle", "reset_next_step"): np.array([base._reset_next_step]),
            ("task", "cumulative_reward"): np.array([cumulative[i]]),
            ("rng", "keys"): keys.astype(np.int64),
            ("rng", "position"): np.array([position], np.int64),
            ("rng", "has_gauss"): np.array([bool(has_gauss)]),
            ("rng", "cached_gaussian"): np.array([cached]),
        }
        for key in lanes._base_fields:
            if key in MODEL_FIELDS:
                field, index = MODEL_FIELDS[key]
                row[key] = np.asarray(getattr(physics.named.model, field)[index], np.float64).reshape(-1).copy()
        if "activation" in lanes._base_fields:
            row["activation"] = physics.data.act.copy()
        rows.append(row)
    return TensorDict(
        {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}, [len(rows)], device="cpu"
    )


def _source_step(lanes, actions, cumulative):
    """Step every lane through dm_control's own ``step``, one lane at a time."""
    rewards, observations, infos = [], [], []
    for i, env in enumerate(lanes.envs):
        reward = 0.0
        for _ in range(2):
            result = env.step(actions[i].astype(env.action_spec().dtype))
            reward += result.reward or 0.0
        cumulative[i] += reward
        rewards.append(reward)
        observations.append(np.concatenate([v.flatten() for v in result.observation.values()], dtype=np.float32))
        infos.append(_source_info(lanes.key, env.physics))
    out = {
        "reward": torch.tensor(np.array(rewards, np.float64)),
        "observation": torch.from_numpy(np.stack(observations)),
        "score": torch.tensor(cumulative / 1000.0),
    }
    if infos[0] is not None:
        out["success"] = torch.tensor([s for s, _ in infos], dtype=torch.float64)
        out["info"] = {key: torch.from_numpy(np.stack([info[key] for _, info in infos])) for key in infos[0][1]}
    return out


def _source_fields(state):
    return state.exclude("physics")


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_batched_step_equals_dm_control_lane_by_lane(key):
    batched = make_local(key, LANES, "cpu", 3)
    source = make_local(key, LANES, "cpu", 3)
    try:
        cumulative = np.zeros(LANES)
        limit = math.ceil(batched._bases[0]._step_limit)
        state = batched.get_state()
        # Three lanes reach the time limit inside the steps: two in an action's second source step, one in its first.
        state["lifecycle", "step_count"][:3] = torch.tensor([[limit - 3], [limit - 4], [limit - 7]])
        batched.set_state(state)
        source.set_state(state)
        assert_same_state(_source_fields(batched.get_state()), _source_state(source, cumulative))
        actions = np.random.default_rng(11).uniform(-1, 1, (STEPS, LANES, batched.action_dim))
        resets = 0
        for t in range(STEPS):
            counts = [base._step_count for base in source._bases]
            result = batched.step(torch.from_numpy(actions[t]), capture_state=t % 3 != 1)
            expected = _source_step(source, actions[t], cumulative)
            for name in ("reward", "observation", "score", "success"):
                assert (name in result.keys()) == (name in expected), name
                if name in expected:
                    assert_same_state(result[name], expected[name])
            if "info" in expected:
                assert set(result["info"].keys()) == set(expected["info"])
                for name, value in expected["info"].items():
                    assert_same_state(result["info", name], value)
            if "state" in result.keys():
                assert_same_state(_source_fields(result["state"]), _source_state(source, cumulative))
            resets += sum(base._step_count < count for base, count in zip(source._bases, counts, strict=True))
        assert_same_state(_source_fields(batched.get_state()), _source_state(source, cumulative))
        assert resets == 3, "every lane near the time limit resets once"
    finally:
        batched.close()
        source.close()


@pytest.mark.parametrize("sigmoid", ["gaussian", "long_tail"])
def test_per_lane_tolerance_equals_dm_control_per_lane(sigmoid):
    from dm_control.utils import rewards

    from stable_envs.envs.dmcontrol.simulation.batch import _tolerance

    x = np.random.default_rng(2).uniform(-0.2, 0.4, 64)
    upper = np.repeat([0.015, 0.02, 0.03, 0.05], 16)
    margin = np.where(np.arange(64) % 3 == 0, 0.0, 5 * upper)
    expected = [rewards.tolerance(v, bounds=(0, u), margin=m, sigmoid=sigmoid) for v, u, m in zip(x, upper, margin)]
    assert_same_state(
        torch.from_numpy(_tolerance(x, 0, upper, margin, sigmoid)), torch.tensor(expected, dtype=torch.float64)
    )


def test_divergent_lane_raises_physics_error_as_dm_control_does():
    from dm_control.rl.control import PhysicsError

    lanes = make_local("cheetah", 2, "cpu", 0)
    try:
        action = np.zeros((2, lanes.action_dim))
        lanes.envs[1].physics.data.qvel[:] = 1e12
        with pytest.raises(PhysicsError) as batched:
            lanes.step(action)
        lanes.envs[0].physics.data.qvel[:] = 1e12
        with pytest.raises(PhysicsError) as source:
            lanes.envs[0].step(action[0])
        assert str(batched.value) == str(source.value)
    finally:
        lanes.close()
