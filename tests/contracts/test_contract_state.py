import pytest

from tests.helpers.reference.bridges import assert_tasks
from tests.helpers.reference.semantics import compare_state

pytestmark = [pytest.mark.integration, pytest.mark.reference]


def test_restore_and_ownership(worlds):
    first, second, source, actions = worlds
    source.compare(first)
    initial = first.get_state()
    assert all(value.device.type == first.device.type for value in initial.values(True, True))
    held_initial = initial.clone()
    for action in actions[:8]:
        first.step(action, states=None, frames=None)
        source.step(action)
    compare_state(initial, held_initial)
    source.compare(first)
    saved = first.get_state()
    frozen = saved.clone()
    for action in actions[8:11]:
        second.step(action, states=None, frames=None)
    second.set_state(saved)
    compare_state(second.get_state(), frozen)
    source.compare(second)
    for value in saved.values(True, True):
        value.zero_()
    compare_state(second.get_state(), frozen)
    malformed = frozen.clone()
    del malformed[next(iter(malformed.keys(True, True)))]
    with pytest.raises((TypeError, ValueError, KeyError)):
        second.set_state(malformed)
    compare_state(second.get_state(), frozen)
    for action in actions[8:16]:
        expected = source.step(action)
        actual = second.step(action, states=None, frames=None)
        source.compare(second)
        assert_tasks(actual, expected, second.device)
    compare_state(initial, held_initial)
