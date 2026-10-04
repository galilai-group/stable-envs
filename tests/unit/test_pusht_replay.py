"""Private replay candidate: ownership, dynamic history and atomic failure.

Source acceptance is a separate independent-oracle gate; these regression tests
do not promote either public PushT registration.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pymunk")
torch = pytest.importorskip("torch")
pytest.importorskip("pygame")
pytest.importorskip("cv2")

from stable_envs._adapters.pusht_numeric import decode_seed, encode_seed, get_value, put_value
from stable_envs._adapters.pusht_replay import PushTReplay
from stable_envs._adapters.pusht_replay_batch import PushTReplayAdapter

pytestmark = pytest.mark.unit


def assert_state(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize(
    "value", [0.25, 40, True, np.float32(0.25), np.int64(2), np.array(1.5), np.array([1, 2, 255], np.uint8)]
)
def test_numeric_representation_round_trip(value):
    state = {}
    put_value(state, "v", value)
    result = get_value(state, "v")
    assert type(result) is type(value)
    np.testing.assert_array_equal(result, value, strict=True)


@pytest.mark.parametrize("seed", [None, -1, 0, 17, 2**127 + 1234, 2**128 - 1])
def test_seed_metadata_preserves_fresh_entropy(seed):
    assert decode_seed(encode_seed(seed)) == seed


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("length", [0, 1, 100, 300, 1025, 10240])
def test_finite_prefix_has_no_storage_horizon(discrete, length):
    env = PushTReplay(discrete=discrete, config={"resolution": 32}, seed=17)
    restored = None
    try:
        for step in range(length):
            action = (
                np.array([step % 16, step // 16 % 16], np.int64)
                if discrete
                else np.array([(-1) ** (step // 32), (-1) ** (step // 64)], np.float32 if step % 2 else np.float64)
            )
            env.step(action)
        snapshot = env.get_state()
        assert all(np.isfinite(value).all() for value in snapshot.values())
        assert snapshot["current/body/agent/mass_infinite"]
        assert snapshot["history/actions"].shape == (length, 2)
        restored = env.reconstruct(snapshot)
        for step in range(10):
            action = np.array([step, 15 - step], np.int32) if discrete else np.array([0.3, -0.2], np.float32)
            a, b = env.step(action), restored.step(action)
            np.testing.assert_array_equal(a[0]["state"], b[0]["state"])
            assert a[1:4] == b[1:4]
        np.testing.assert_array_equal(env.env.render(), restored.env.render())
        assert snapshot["history/actions"].shape == (length, 2)
    finally:
        env.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("discrete", [False, True])
def test_reset_retains_action_marker_and_owned_rng(discrete):
    env = PushTReplay(discrete=discrete, config={"resolution": 96, "render_action": True}, seed=None)
    reset = restored = None
    try:
        env.step(np.array([1, 0], np.int64 if discrete else np.float32))
        # Own both lazy Gym generators before capturing the fresh reset recipe.
        env.env.np_random.random()
        env.env.action_space.sample()
        reset = env.reset_candidate(None, {"variation": ["all"]})
        restored = reset.reconstruct(reset.get_state())
        np.testing.assert_array_equal(reset.env.latest_action, env.env.latest_action)
        np.testing.assert_array_equal(reset.env._goal, restored.env._goal)
        np.testing.assert_array_equal(reset.env.render(), restored.env.render())
        np.testing.assert_array_equal(reset.env.action_space.sample(), restored.env.action_space.sample())
        assert reset.env.np_random.random() == restored.env.np_random.random()
    finally:
        for candidate in (env, reset, restored):
            if candidate is not None:
                candidate.close()


@pytest.mark.parametrize("discrete", [False, True])
def test_batch_padding_active_lanes_ownership_and_restore(discrete):
    env = PushTReplayAdapter(16, discrete=discrete, config={"resolution": 32})
    try:
        action = np.ones((16, 2), np.int64 if discrete else np.float32)
        env.step(action)
        owned = env.get_state()
        for i in range(4):
            env.step(action, active=np.arange(16) % 4 <= i)
        saved = env.get_state()
        assert saved["history", "actions"].shape == (16, 5, 2)
        assert torch.equal(saved["history", "lengths"], torch.tensor([5, 4, 3, 2] * 4))
        assert owned["history", "actions"].shape == (16, 1, 2)
        expected = env.step(action)["state"]
        env.set_state(saved)
        assert_state(env.step(action)["state"], expected)
        saved["history", "actions"].zero_()
        assert_state(env.get_state(), expected)
    finally:
        env.close()


@pytest.mark.parametrize("damage", ["late_diagnostic", "dtype", "padding", "mask", "rng", "version", "variation"])
def test_invalid_state_never_partially_installs_lanes(damage):
    env = PushTReplayAdapter(2, config={"resolution": 32})
    try:
        env.step(np.ones((2, 2), np.float64), active=np.array([True, False]))
        before = env.get_state()
        bad = before.clone()
        if damage == "late_diagnostic":
            bad["current", "body", "block", "position"][1, 0] += 1
        elif damage == "dtype":
            bad["history", "dtypes"][0, 0] = 3
        elif damage == "padding":
            bad["history", "actions"][1, 0, 0] = 1
        elif damage == "mask":
            bad["history", "valid"][1, 0] = True
        elif damage == "rng":
            bad["anchor", "rng"][1, 5] = 0
        elif damage == "version":
            bad["recipe", "version"][1] += 1
        else:
            bad["owned", "variation", "block", "shape", "current", "value"][1] = 100
        with pytest.raises(ValueError):
            env.set_state(bad)
        assert_state(env.get_state(), before)
    finally:
        env.close()


@pytest.mark.parametrize("discrete", [False, True])
def test_invalid_last_action_and_failed_reset_are_atomic(discrete):
    env = PushTReplayAdapter(2, discrete=discrete, config={"resolution": 32})
    try:
        before = env.get_state()
        action = np.ones((2, 2), np.int64 if discrete else np.float64)
        action[-1, -1] = 16 if discrete else np.nan
        with pytest.raises(ValueError):
            env.step(action)
        assert_state(env.get_state(), before)
        with pytest.raises(ValueError):
            env.reset(17, {"goal_state": np.full(7, np.nan)})
        assert_state(env.get_state(), before)
    finally:
        env.close()


def test_success_does_not_impose_an_auto_reset_or_permanent_stop():
    start = np.array([250, 220, 250, 250, 0, 0, 0], np.float64)
    env = PushTReplayAdapter(1, config={"resolution": 32})
    try:
        env.reset(17, {"variation": [], "state": start, "goal_state": start})
        assert env.step(np.zeros((1, 2), np.float64))["terminated"].item()
        before = env.get_state()
        env.step(np.ones((1, 2), np.float64))
        assert env.get_state()["history", "lengths"].item() == 2
        assert not torch.equal(before["current", "observation"], env.get_state()["current", "observation"])
    finally:
        env.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bad", [-1.01, 1.01, np.inf, -np.inf, np.nan])
def test_continuous_bounds_reject_atomically(dtype, bad):
    env = PushTReplayAdapter(2, config={"resolution": 32})
    try:
        before = env.get_state()
        actions = np.zeros((2, 2), dtype)
        actions[1, 1] = bad
        with pytest.raises(ValueError):
            env.step(actions)
        assert_state(env.get_state(), before)
    finally:
        env.close()
