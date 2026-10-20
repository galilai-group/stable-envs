"""Full-State, action validation and lifecycle contracts for G34 nominal."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as se
from tests.helpers.reference.semantics import (
    assert_lossy_projection_rejected,
    assert_same_state,
    check_two_histories_same_state,
)


def test_restoration_after_other_history_keeps_hidden_ball_and_owns_storage():
    with se.MuJoCoOcclusionBall(num_envs=2) as first, se.MuJoCoOcclusionBall(num_envs=2) as second:
        zero = np.zeros((2, 1), np.float64)
        first.step(np.zeros((2, 59, 1)), frames=None)
        second.step(np.zeros((2, 8, 1)), frames=None)
        saved = first.get_state()
        held = saved.clone()
        check_two_histories_same_state(first, second, saved, [zero] * 20)
        assert_same_state(saved, held)
        assert_lossy_projection_rejected(first, saved)
        expected = first.get_state()
        saved["physics", "integration"].zero_()
        assert_same_state(first.get_state(), expected)


@pytest.mark.parametrize("error", ["dtype", "extra", "missing", "frame", "clock", "quaternion", "nan", "batch"])
def test_bad_state_is_rejected_before_any_lane_changes(error):
    with se.MuJoCoOcclusionBall(num_envs=2) as env:
        before = env.get_state()
        bad = before.clone()
        if error == "dtype":
            bad["physics", "integration"] = bad["physics", "integration"].float()
        elif error == "extra":
            bad["extra"] = torch.zeros(2, 1)
        elif error == "missing":
            del bad["lifecycle"]
        elif error == "frame":
            bad["lifecycle", "elapsed_steps"][1] = 120
        elif error == "clock":
            bad["physics", "integration"][1, 0] = 1
        elif error == "quaternion":
            bad["physics", "integration"][1, -4:] = 0
        elif error == "nan":
            bad["physics", "integration"][1, 0] = float("nan")
        else:
            bad = bad[:1]
        with pytest.raises((ValueError, TypeError)):
            env.set_state(bad)
        assert_same_state(env.get_state(), before)


@pytest.mark.parametrize(
    "actions", [np.ones((2, 1)), np.zeros((2, 1), np.int64), np.zeros((1, 1)), np.full((2, 1), np.nan)]
)
def test_actions_are_explicit_noops_and_invalid_actions_are_atomic(actions):
    with se.MuJoCoOcclusionBall(num_envs=2) as env:
        before = env.get_state()
        with pytest.raises(ValueError):
            env.step(actions, frames=None)
        assert_same_state(env.get_state(), before)


def test_finished_lanes_hold_their_own_endpoint_in_a_sequence():
    with se.MuJoCoOcclusionBall(num_envs=2) as env:
        env.step(np.zeros((2, 118, 1)), frames=None)
        state = env.get_state()
        state["lifecycle", "elapsed_steps"][1] = 117
        state["physics", "integration"][1, 0] = 117 / 24
        env.set_state(state)
        out = env.step(np.zeros((2, 4, 1)), frames=None)
        assert out["valid"].tolist() == [[True, False, False, False], [True, True, False, False]]
        assert not bool(out["terminated"].any())
        assert out["truncated"].tolist() == [[True, False, False, False], [False, True, False, False]]
        assert out["states"]["lifecycle", "elapsed_steps"][:, :, 0].tolist() == [[119] * 4, [118, 119, 119, 119]]
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros((2, 1)), frames=None)


def test_reset_seed_is_a_noop_and_unsupported_configuration_is_explicit():
    with se.MuJoCoOcclusionBall() as env:
        assert_same_state(env.reset(seed=3), env.reset(seed=99))
        with pytest.raises(ValueError):
            env.reset(options={"ignored": True})
    for options in ({"size": 84}, {"backend": "triton"}, {"device": "cuda"}, {"num_envs": 0}):
        with pytest.raises(ValueError):
            se.MuJoCoOcclusionBall(**options)
