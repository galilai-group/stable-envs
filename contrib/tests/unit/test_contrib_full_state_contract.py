"""The ``MiniGridEmpty`` cases of ``tests/unit/test_full_state_contract.py``, run through the same test bodies."""

from __future__ import annotations

import pytest

from tests.unit import test_full_state_contract as _main


@pytest.mark.parametrize("name", ["MiniGridEmpty"])
def test_full_snapshot_restores_future_and_has_no_aliases(name):
    _main.test_full_snapshot_restores_future_and_has_no_aliases(name)


@pytest.mark.parametrize("name", ["MiniGridEmpty"])
def test_lossy_projection_is_rejected_and_full_state_renders(name):
    _main.test_lossy_projection_is_rejected_and_full_state_renders(name)
