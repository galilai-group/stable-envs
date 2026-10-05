"""Exact source task outputs; family boundary witnesses remain separate gates."""

import pytest

from tests.helpers.reference.bridges import assert_tasks

pytestmark = [pytest.mark.integration, pytest.mark.reference]


def test_independent_source_task_outputs(worlds):
    candidate, _, source, actions = worlds
    for action in actions:
        expected = source.step(action)
        actual = candidate.step(action, states=None, frames=None)
        assert_tasks(actual, expected, candidate.device)
