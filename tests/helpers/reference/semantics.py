"""Shared full-State parity checks, independent of pixels and simulator family.

Reference bridges normalize authoritative fields to TensorDict; they must not
instantiate the candidate adapter or use its serialization as their oracle.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from stable_envs.testing.metrics import leaf_difference


def _step_without_pixels(env, action):
    from stable_envs.env import Env

    return env.step(action, states=None, frames=None).exclude("valid") if isinstance(env, Env) else env.step(action)


def compare_state(actual: TensorDict, expected: TensorDict, *, atol=0.0) -> dict:
    """Compare every leaf, including RNG/lifecycle fields, with explicit error evidence."""
    ak = set(actual.keys(True, True))
    ek = set(expected.keys(True, True))
    assert ak == ek, f"State keys differ: {ak ^ ek}"
    assert actual.batch_size == expected.batch_size
    errors = {}
    for key in sorted(ak, key=str):
        a, e = actual[key].detach().cpu(), expected[key].detach().cpu()
        diagnostic = leaf_difference(a, e)
        assert diagnostic["comparable"], f"{key}: {diagnostic}"
        # Tolerances belong to continuous values, never counters, masks or RNG.
        tolerance = atol if a.is_floating_point() else 0
        torch.testing.assert_close(
            a, e, atol=tolerance, rtol=0, equal_nan=True, msg=lambda m: f"{key}: {diagnostic}\n{m}"
        )
        errors[str(key)] = diagnostic["max_abs_error"]
    return errors


def check_reset_matches_reference(candidate, reference, *, seed, atol=0.0):
    return compare_state(candidate.reset(seed=seed), reference.reset(seed=seed), atol=atol)


def check_state_roundtrip(env, *, actions=()):
    saved = env.get_state()
    for action in actions:
        _step_without_pixels(env, action)
    env.set_state(saved)
    return compare_state(env.get_state(), saved)


def check_reward_done_matches_reference(actual, expected, *, atol=0.0):
    keys = ("reward", "terminated", "truncated")
    for key in keys:
        assert key in actual and key in expected, f"Missing {key}"
    return compare_state(actual.select(*keys), expected.select(*keys), atol=atol)


def check_step_matches_reference(candidate, reference, actions, *, atol=0.0):
    """Caller first establishes equivalent full State and persistent Variation."""
    evidence = []
    for action in actions:
        actual, expected = _step_without_pixels(candidate, action), reference.step(action)
        state = compare_state(actual["state"], expected["state"], atol=atol)
        task = check_reward_done_matches_reference(actual, expected, atol=atol)
        outputs = compare_state(actual.exclude("state"), expected.exclude("state"), atol=atol)
        evidence.append(dict(state=state, task=task, outputs=outputs))
    return evidence


def check_two_histories_same_state(first, second, state, actions, *, atol=0.0):
    """Instances must arrive from different histories, then resume the same snapshot."""
    first.set_state(state)
    second.set_state(state)
    compare_state(first.get_state(), second.get_state(), atol=atol)
    return [compare_state(_step_without_pixels(first, a), _step_without_pixels(second, a), atol=atol) for a in actions]
