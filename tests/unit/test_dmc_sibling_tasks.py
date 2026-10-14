"""The DM Control cells simulate dm_control's own registered tasks exactly."""

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

import stable_envs as se  # noqa: E402
from tests.helpers.reference.dmc_suite import SuiteReference  # noqa: E402
from tests.helpers.reference.semantics import compare_state  # noqa: E402

CELLS = {
    "DMCCartpoleBalance": ("cartpole", "balance"),
    "DMCFingerSpin": ("finger", "spin"),
    "DMCFingerTurnEasy": ("finger", "turn_easy"),
    "DMCPointMassEasy": ("point_mass", "easy"),
    "DMCReacherEasy": ("reacher", "easy"),
    "DMCSwimmerSwimmer6": ("swimmer", "swimmer6"),
}
#: SWM-registered cells against the suite task each simulates. The aligned cells (SWM's MJCF round
#: trip) and walker (SWM's task reward) replay corpora instead, in test_dmc_oracle_replay.py.
SWM_CELLS = {
    "DMCAcrobotBalance": ("acrobot", "swingup_sparse"),
    "DMCPendulumSwingup": ("pendulum", "swingup"),
    "DMCCartpoleSwingup": ("cartpole", "swingup_sparse"),
    "DMCCheetahRun": ("cheetah", "run"),
    "DMCHopperHop": ("hopper", "hop"),
    "DMCHumanoidWalk": ("humanoid", "walk"),
    "DMCReacherHard": ("reacher", "hard"),
}
TRACKCOM = {"DMCSwimmerSwimmer6", "DMCCheetahRun", "DMCHopperHop", "DMCHumanoidWalk"}


def _make(name):
    # These render through MJWarp; their simulation does not need the renderer.
    return se.make(name, num_envs=2, device="cpu", **({"backend": "opengl"} if name in TRACKCOM else {}))


def _source_fields(state):
    return state.exclude("physics")


def _rollout(env, ref, actions):
    for action in actions:
        result = env.step(action, frames=None)
        expected, reward, observation = ref.step(action)
        compare_state(_source_fields(result["state"]), expected)
        assert torch.equal(result["reward"], reward)
        assert torch.equal(result["observation"], observation)


@pytest.mark.parametrize("name", CELLS)
def test_cell_matches_the_source_task_and_restores_into_a_fresh_env(name):
    domain, task = CELLS[name]
    seed = 5
    ref = SuiteReference(domain, task, 2)
    try:
        with _make(name) as env:
            actions = np.random.default_rng(seed).uniform(-1, 1, (24, 2, env.action_dim))
            compare_state(_source_fields(env.reset(seed=seed)), ref.reset(seed))
            _rollout(env, ref, actions[:8])
            saved = env.get_state()
            # Both lanes cross the source's time limit inside the next actions.
            near_limit = saved.clone()
            near_limit["lifecycle", "step_count"][:] = ref.step_limit - 3
            for wrapper in ref.envs:
                wrapper._env._step_count = ref.step_limit - 3
            env.set_state(near_limit)
            compare_state(env.get_state(), near_limit)
            _rollout(env, ref, actions[8:16])
            resumed = env.get_state()
            assert bool((resumed["lifecycle", "step_count"] < ref.step_limit - 3).all()), "no implicit reset"
            with _make(name) as fresh:
                fresh.set_state(resumed)
                compare_state(fresh.get_state(), resumed)
                _rollout(fresh, ref, actions[16:])
                env.set_state(fresh.get_state())
                compare_state(env.get_state(), fresh.get_state())
    finally:
        ref.close()


@pytest.mark.parametrize("name", SWM_CELLS)
def test_swm_cell_matches_the_suite_task(name):
    seed = 7
    ref = SuiteReference(*SWM_CELLS[name], 2)
    try:
        with _make(name) as env:
            actions = np.random.default_rng(seed).uniform(-1, 1, (24, 2, env.action_dim))
            compare_state(_source_fields(env.reset(seed=seed)), ref.reset(seed))
            _rollout(env, ref, actions)
    finally:
        ref.close()
