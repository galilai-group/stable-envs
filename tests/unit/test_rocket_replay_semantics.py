"""Official-engine Rocket State ownership and atomicity regressions."""

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")

from stable_envs import make


def exact(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    assert all(torch.equal(a[k], b[k]) for k in a.keys(True, True))


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    a, b = make("PyFlytRocketLanding", num_envs=2), make("PyFlytRocketLanding", num_envs=2, seed=42)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def test_whole_batch_invalid_action_does_not_step_first_lane(pair):
    a, b = pair
    b.set_state(a.get_state())
    saved = a.get_state()
    actions = np.zeros((2, 7), np.float64)
    actions[1, 5] = 2
    for _ in range(3):
        with pytest.raises(AssertionError):
            a.step(actions)
    exact(saved, a.get_state())
    exact(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


@pytest.mark.parametrize(
    "kind", ["nan", "missing", "dtype", "history_mask", "config", "unreachable_second_lane", "rng_increment"]
)
def test_invalid_whole_state_is_atomic(pair, kind):
    a, b = pair
    original = a.get_state()
    corrupted = original.clone()
    if kind == "nan":
        corrupted["current", "wrapper", "reward"][1] = float("nan")
    elif kind == "missing":
        del corrupted["history", "length"]
    elif kind == "dtype":
        corrupted["history", "actions"] = corrupted["history", "actions"].float()
    elif kind == "history_mask":
        corrupted["history", "valid"][1, 0] = True
    elif kind == "config":
        corrupted["recipe", "config"][1, 0] += 1
    elif kind == "unreachable_second_lane":
        corrupted["current", "wrapper", "reward"][1] += 1
    elif kind == "rng_increment":
        corrupted["recipe", "main_rng"][1, 5] = 2
    with pytest.raises((ValueError, TypeError)):
        a.set_state(corrupted)
    exact(original, a.get_state())
    b.set_state(original)
    exact(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


def test_independent_storage_mixed_dtypes_and_different_histories(pair):
    a, b = pair
    first = a.get_state()
    for i in range(12):
        a.step(np.array([[0.2, -0.3, 0.1, 1, 0.4, 0.1, -0.2]] * 2, dtype=np.float32 if i % 2 else np.float64))
    saved = a.get_state()
    a.set_state(first)
    b.set_state(saved)
    assert not torch.equal(a.get_state()["history", "length"], saved["history", "length"])
    a.set_state(saved)
    saved["current", "wrapper", "reward"].fill_(123)
    for _ in range(3):
        exact(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))
    assert a.get_state()["history", "length"].dtype == torch.int64
    exact(a.to_render_state(), b.to_render_state())


def test_seeded_future_reset_and_fresh_entropy(pair):
    a, b = pair
    a.reset(seed=17, options={"variation": ["all"]})
    b.set_state(a.get_state())
    a.set_variation({"environment.start_tilt": np.array([0.1, -0.1], np.float32)})
    b.set_state(a.get_state())
    exact(a.reset(seed=42, options={"variation": ["rocket"]}), b.reset(seed=42, options={"variation": ["rocket"]}))
    a.reset(seed=None, options={"variation": ["all"]})
    saved = a.get_state()
    b.set_state(saved)
    exact(saved, b.get_state())
    a.reset(seed=None, options={"variation": ["all"]})
    fresh = a.get_state()
    assert not torch.equal(
        saved["variation", "rocket", "body_color", "rng"], fresh["variation", "rocket", "body_color", "rng"]
    )
    b.set_state(fresh)
    exact(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


def test_cpu_render_explicitly_unsupported(pair):
    with pytest.raises(NotImplementedError, match="no CPU"):
        pair[0].render()
