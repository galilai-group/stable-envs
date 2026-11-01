"""TwoRoom JAX physics and shared task outputs against pinned SWM."""

from contextlib import closing

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.semantics import compare_state

pytestmark = [pytest.mark.family("navigation"), pytest.mark.swm]


@pytest.mark.parametrize("backend", ["jax", "numba"])
@pytest.mark.parametrize("seed", [0, 7, 31])
def test_public_steps_match_swm(seed, backend):
    with (
        closing(NativeNavigationReference("two_room", 8)) as source,
        se.TwoRoom(num_envs=8, device="cpu", simulation_backend=backend) as env,
    ):
        compare_state(env.reset(seed=seed), source.reset(seed=seed))
        assert env.execution_mode == {"jax": "NATIVE_JAX", "numba": "NATIVE_NUMBA"}[backend]
        actions = np.random.default_rng(seed).uniform(-2, 2, (40, 8, 2)).astype(np.float32)
        for action in actions:
            compare_state(env.step(action, states=None, frames=None).exclude("valid"), source.step(action))


@pytest.mark.parametrize("backend", ["jax", "numba"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("doors", [1, 2, 3])
def test_collisions_and_observations_match_swm_with_variation(axis, doors, backend):
    values = {
        "agent.radius": [7.125],
        "agent.speed": [5.5],
        "wall.axis": axis,
        "wall.thickness": 11,
        "door.number": doors,
        "door.position": [49, 112, 180],
        "door.size": [14, 10, 7],
    }
    positions = [[95, 100], [130, 100], [100, 49], [50, 21], [202, 100], [49, 100], [100, 130]]
    actions = torch.tensor([[1, 0], [-1, 0], [1, 0], [0, -5], [5, 0], [0, 1], [0, -1]], dtype=torch.float64)
    with (
        closing(NativeNavigationReference("two_room", 7, persistent_values=values)) as source,
        se.TwoRoom(num_envs=7, device="cpu", simulation_backend=backend) as env,
    ):
        env.set_variation(values)
        state = env.reset(seed=0, options={"state": positions, "target_state": [180, 180]})
        source.reset(seed=0)
        source.set_state(state)
        for _ in range(6):
            compare_state(env.step(actions, states=None, frames=None).exclude("valid"), source.step(actions))


def test_success_boundary_matches_swm():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    targets = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    with (
        closing(NativeNavigationReference("two_room", len(targets))) as source,
        se.TwoRoom(num_envs=len(targets), device="cpu") as env,
    ):
        state = env.reset(seed=0, options={"state": [60, 60], "target_state": targets})
        source.reset(seed=0)
        source.set_state(state)
        action = torch.zeros(len(targets), 2)
        compare_state(env.step(action, states=None, frames=None).exclude("valid"), source.step(action))


def test_continuous_action_rounding_matches_swm():
    options = {"state": [53.47578811645508, 60], "target_state": [180, 180]}
    action = torch.tensor([[-0.9868243932723999, 0.0]])
    with (
        closing(NativeNavigationReference("two_room", options=options)) as source,
        se.TwoRoom(device="cpu") as env,
    ):
        compare_state(env.reset(seed=0, options=options), source.reset(seed=0))
        compare_state(env.step(action, states=None, frames=None).exclude("valid"), source.step(action))
