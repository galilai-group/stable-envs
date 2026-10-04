"""Independent source fields along a reached trajectory, including reset."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.reference]


def test_independent_source_trajectory(worlds):
    candidate, _, source, actions = worlds
    source.compare(candidate)
    for action in actions:
        candidate.step(action, states=None, frames=None)
        source.step(action)
        source.compare(candidate)
