"""DMC process lanes match local lanes for every worker count, and a rejected State restores no lane."""

import numpy as np
import pytest

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import make_local  # noqa: E402
from stable_envs.envs.dmcontrol.simulation.process import DMCProcessAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402


@pytest.mark.parametrize("domain", ["cartpole", "reacher", "walker", "finger-turn_easy"])
def test_process_matches_local(domain):
    other = make_local(domain, 5, seed=99)
    try:
        other.step(np.full((5, other.action_dim), 0.2))
        foreign = other.get_state()
        actions = list(np.random.default_rng(5).uniform(-1, 1, (8, 5, other.action_dim)))
    finally:
        other.close()
    assert_process_matches_local(
        lambda: make_local(domain, 5, seed=17),
        lambda count: DMCProcessAdapter(domain, 5, seed=17, processes=count),
        actions,
        counts=(1, 2, 5),
        foreign=foreign,
    )


def test_invalid_last_lane_restores_no_lane():
    adapter = DMCProcessAdapter("cartpole", 5, seed=17, processes=2)
    try:

        def corrupt(state):
            state["qpos"][0] += 0.1
            state["qvel"][-1, 0] = float("nan")
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(np.zeros((5, 1))))
    finally:
        adapter.close()
