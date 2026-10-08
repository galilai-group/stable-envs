"""Exact comparisons between candidate Fetch State/outputs and pinned source environments.

Copied unchanged from the historical public Fetch gate and adapter probe
(pre-repository-cleanup:scripts/recovery/fetch_public_gate.py and
fetch_pickandplace_adapter_probe.py). Values, shapes and dtypes are compared exactly.
"""

import gymnasium as gym
import numpy as np
import torch


def oracle(envs):
    """Read the source fields that candidate Fetch State must reproduce, one row per lane."""
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
    for lane, r in enumerate(expected):
        row = state[lane].cpu()
        for f in ("qpos", "qvel", "warmstart", "goal"):
            assert np.array_equal(row[f].numpy(), r[f]), (lane, f, np.max(abs(row[f].numpy() - r[f])))
        for f in ("position", "quaternion"):
            assert np.array_equal(row["controller", f].numpy(), r[f]), (lane, f)
        assert row["lifecycle", "elapsed_steps"].item() == r["elapsed"]
        for field, source in [("state_words", "state"), ("increment_words", "inc")]:
            value = sum(int(x) << (32 * i) for i, x in enumerate(row["rng", field]))
            assert value == r["rng"]["state"][source]
        for f in ("has_uint32", "uinteger"):
            assert row["rng", f].item() == r["rng"][f]


def exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, "shape", actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (label, "dtype", actual.dtype, expected.dtype)
    assert np.array_equal(actual, expected), label


def full_compare(state, refs):
    """Compare every State field, simulation time and reset anchor, plus declared shapes and dtypes."""
    compare_state(state, oracle(refs))
    for lane, e in enumerate(refs):
        exact_array(state["time"][lane].numpy(), np.array([e.unwrapped.data.time]), "time")
        exact_array(
            state["task", "initial_gripper_xpos"][lane].numpy(), e.unwrapped.initial_gripper_xpos, "reset_anchor"
        )
    from stable_envs.envs.fetch.simulation.adapter import state_fields

    fields = state_fields(refs[0].unwrapped.model.nq, refs[0].unwrapped.model.nv)
    assert set(state.keys(True, True)) == set(fields)
    for key, (width, dtype) in fields.items():
        assert state[key].shape == (len(refs), width) and state[key].dtype == dtype, key


def equal_state(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for k in a.keys(True, True):
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k
