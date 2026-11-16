"""Exact comparisons between candidate Fetch State/outputs and pinned source environments."""

import gymnasium as gym
import numpy as np

from stable_envs import get_field
from stable_envs._core.state_schema import unfold


def oracle(envs):
    """Read the source fields that candidate Fetch State must reproduce, one row per environment."""
    result = []
    for e in envs:
        u = e.unwrapped
        while not isinstance(e, gym.wrappers.TimeLimit):
            e = e.env
        r = u.np_random.bit_generator.state
        i = u.model.body("robot0:gripper_link").id
        result.append(
            {
                "qpos": u.data.qpos.copy(),
                "qvel": u.data.qvel.copy(),
                "warmstart": u.data.qacc_warmstart.copy(),
                "position": u.data.xpos[i].copy(),
                "quaternion": u.data.xquat[i].copy(),
                "goal": u.goal.copy(),
                "elapsed": e._elapsed_steps,
                "rng": r,
            }
        )
    return result


def compare_state(state, expected):
    rows = unfold(state.cpu())
    for instance, r in enumerate(expected):
        row = rows[instance]
        for f in ("qpos", "qvel", "warmstart", "goal"):
            got = get_field(row, f).numpy()
            assert np.array_equal(got, r[f]), (instance, f, np.max(abs(got - r[f])))
        for f in ("position", "quaternion"):
            assert np.array_equal(get_field(row, f"controller/{f}").numpy(), r[f]), (instance, f)
        assert get_field(row, "lifecycle/elapsed_steps").item() == r["elapsed"]
        for field, source in [("state_words", "state"), ("increment_words", "inc")]:
            value = sum(int(x) << (32 * i) for i, x in enumerate(get_field(row, f"rng/{field}")))
            assert value == r["rng"]["state"][source]
        for f in ("has_uint32", "uinteger"):
            assert get_field(row, f"rng/{f}").item() == r["rng"][f]


def exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, "shape", actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (label, "dtype", actual.dtype, expected.dtype)
    assert np.array_equal(actual, expected), label


def full_compare(state, refs):
    """Compare every State field, simulation time and reset anchor, plus declared shapes and dtypes."""
    compare_state(state, oracle(refs))
    for instance, e in enumerate(refs):
        exact_array(get_field(state, "time")[instance].numpy(), np.array([e.unwrapped.data.time]), "time")
        exact_array(
            get_field(state, "task/initial_gripper_xpos")[instance].numpy(),
            e.unwrapped.initial_gripper_xpos,
            "reset_anchor",
        )
    from stable_envs.envs.fetch.simulation.adapter import fetch_schema

    schema = fetch_schema(refs[0].unwrapped.model.nq, refs[0].unwrapped.model.nv)
    schema.validate(state)
    for name, where in schema.slices.items():
        value = get_field(state, name)
        assert value.shape == (len(refs), *where.shape) and value.dtype == where.dtype, name
