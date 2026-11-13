"""Checks that a process adapter gives the same State and task outputs as its local adapter."""

from __future__ import annotations

import pytest

from tests.helpers.reference.semantics import assert_same_state


def exercise(adapter, actions, masks=None, foreign=None):
    """Return every output of a fixed request sequence on ``adapter``.

    The sequence steps with and without State capture, restores a mid-run
    snapshot and repeats the same actions, and checks get/set round trips.
    ``foreign`` is a State from another instance to restore and continue from.
    """

    def step(t):
        kwargs = {} if masks is None else {"active": masks[t]}
        return adapter.step(actions[t], capture_state=t % 3 != 1, **kwargs)

    outputs = [adapter.get_state()]
    half = len(actions) // 2
    for t in range(half):
        outputs.append(step(t))
    middle = adapter.get_state()
    first = [step(t) for t in range(half, len(actions))]
    outputs += first + [adapter.get_state()]
    adapter.set_state(middle)
    assert_same_state(adapter.get_state(), middle)
    end = outputs[-1]
    again = [step(t) for t in range(half, len(actions))]
    # Inactive lanes repeat the previous task outputs, which State does not hold; compare State only.
    for expected, actual in zip(first, again, strict=True):
        assert ("state" in expected.keys()) == ("state" in actual.keys())
        if "state" in expected.keys():
            assert_same_state(actual["state"], expected["state"])
    assert_same_state(adapter.get_state(), end)
    snapshot = adapter.get_state()
    adapter.set_state(snapshot)
    assert_same_state(adapter.get_state(), snapshot)
    if foreign is not None:
        adapter.set_state(foreign)
        assert_same_state(adapter.get_state(), foreign)
        outputs += [step(t) for t in range(half)]
    return outputs


def assert_process_matches_local(make_local, make_process, actions, *, masks=None, counts=(1, 2, 3), foreign=None):
    """Run ``exercise`` on the local adapter and on a process adapter per worker count; all outputs must match."""
    local = make_local()
    try:
        expected = exercise(local, actions, masks, foreign)
    finally:
        local.close()
    for count in counts:
        remote = make_process(count)
        try:
            actual = exercise(remote, actions, masks, foreign)
        finally:
            remote.close()
        assert len(actual) == len(expected)
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            try:
                assert_same_state(got, want)
            except AssertionError as error:
                raise AssertionError(f"processes={count}, output {index}: {error}") from error


def assert_atomic_restore(adapter, corrupt, error, step):
    """A State that ``corrupt`` makes invalid in its last lane raises ``error`` and changes no lane."""
    before = adapter.get_state()
    with pytest.raises(error):
        adapter.set_state(corrupt(before.clone()))
    assert_same_state(adapter.get_state(), before)
    step(adapter)
    adapter.set_state(before)
    assert_same_state(adapter.get_state(), before)
