"""MSHumanLocomotion resumes a trajectory bit-for-bit from a State saved mid-rollout."""

from __future__ import annotations

import numpy as np

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

LANES, SAVE_AT, AFTER, SEED = 3, 5, 10, 7


def test_set_state_resumes_the_trajectory_bit_for_bit():
    rng = np.random.default_rng(SEED)
    with se.make("MSHumanLocomotion", num_envs=LANES, device="cpu") as env:
        actions = [env.action_spec.sample(LANES, rng) for _ in range(SAVE_AT + AFTER)]
        env.reset(seed=SEED)
        for action in actions[:SAVE_AT]:
            env.step(action, frames=None)
        saved = env.get_state()
        expected = [env.step(action, frames=None)["state"] for action in actions[SAVE_AT:]]
        env.set_state(saved)
        for action, want in zip(actions[SAVE_AT:], expected):
            assert_same_state(env.step(action, frames=None)["state"], want)
