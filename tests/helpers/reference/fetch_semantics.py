"""Independent pinned Fetch snapshots and public-State field extraction."""

import copy

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


def snapshot(env):
    """Read the independent source's continuation fields, without a library adapter."""
    raw = env.unwrapped
    limit = env
    while not isinstance(limit, gym.wrappers.TimeLimit):
        limit = limit.env
    gripper = raw.model.body("robot0:gripper_link").id
    return dict(
        {
            name: getattr(raw.data, name).copy()
            for name in (
                "qpos",
                "qvel",
                "qacc_warmstart",
                "mocap_pos",
                "mocap_quat",
                "ctrl",
                "qfrc_applied",
                "xfrc_applied",
            )
        },
        time=np.asarray(raw.data.time),
        goal=raw.goal.copy(),
        initial_gripper_xpos=raw.initial_gripper_xpos.copy(),
        elapsed=np.asarray(limit._elapsed_steps),
        max_episode_steps=np.asarray(limit._max_episode_steps),
        rng=copy.deepcopy(raw.np_random.bit_generator.state),
        controller_position=raw.data.xpos[gripper].copy(),
        controller_quaternion=raw.data.xquat[gripper].copy(),
    )


def differences(a, b, prefix=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        result = {}
        for key in a:
            result.update(differences(a[key], b[key], prefix + "/" + str(key)))
        return result
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        result = {}
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            result.update(differences(x, y, prefix + "/" + str(i)))
        return result
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (prefix, a.shape, b.shape, a.dtype, b.dtype)
    if np.array_equal(a, b):
        return {}
    delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.dtype.kind in "bifu" else None
    return {prefix: dict(max_abs=delta, shape=list(a.shape), dtype=str(a.dtype))}


def source_state(rows):
    """Independent extraction from source snapshots, never an adapter export."""
    columns = {}
    for key, source in [
        ("qpos", "qpos"),
        ("qvel", "qvel"),
        ("time", "time"),
        ("warmstart", "qacc_warmstart"),
        ("goal", "goal"),
        (("task", "initial_gripper_xpos"), "initial_gripper_xpos"),
        (("controller", "position"), "controller_position"),
        (("controller", "quaternion"), "controller_quaternion"),
        (("lifecycle", "elapsed_steps"), "elapsed"),
    ]:
        columns[key] = torch.from_numpy(np.stack([np.atleast_1d(row[source]) for row in rows]))
    for key, field in [("state_words", "state"), ("increment_words", "inc")]:
        words = [
            np.frombuffer(int(row["rng"]["state"][field]).to_bytes(16, "little"), dtype="<u4").astype(np.int64)
            for row in rows
        ]
        columns["rng", key] = torch.from_numpy(np.stack(words))
    for field in ("has_uint32", "uinteger"):
        columns["rng", field] = torch.tensor([[row["rng"][field]] for row in rows], dtype=torch.int64)
    assert all(row["rng"]["bit_generator"] == "PCG64" for row in rows)
    return TensorDict(columns, batch_size=[len(rows)], device="cpu")
