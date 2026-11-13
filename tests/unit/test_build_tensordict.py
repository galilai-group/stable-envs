"""``build_tensordict`` gives what the TensorDict constructor gives, and falls back to it when a leaf does not fit."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._core.tensordicts import build_tensordict
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit


def _layout(td):
    return (
        td.batch_size,
        td.device,
        td.names if td._has_names() else None,
        {key: (v.shape, v.dtype, v.device) for key, v in td.items(True, True)},
        {key: (type(v), v.batch_size, v.device) for key, v in td.items(True) if not torch.is_tensor(v)},
    )


def _source():
    nested = TensorDict({"x": torch.arange(6).reshape(3, 1, 2)}, batch_size=[3, 1], device="cpu")
    return {
        "a": torch.zeros(3, 4),
        "tree": {"b": torch.ones(3, dtype=torch.int32), "deeper": {"c": torch.arange(3)}},
        "td": nested,
    }


def test_a_fitting_source_builds_the_same_tensordict_and_keeps_the_leaves():
    built = build_tensordict(_source(), [3], "cpu")
    reference = TensorDict(_source(), batch_size=[3], device="cpu")
    assert _layout(built) == _layout(reference)
    source = _source()
    built = build_tensordict(source, [3], torch.device("cpu"))
    assert built["a"] is source["a"] and built["td"] is source["td"]
    assert_same_state(built, TensorDict(_source(), batch_size=[3], device="cpu"))


@pytest.mark.parametrize(
    "source",
    [
        {"a": torch.zeros(3, 2, dtype=torch.float64), "n": np.arange(3)},
        {("x", "y"): torch.zeros(3)},
        {"a": TensorDict({"b": torch.zeros(3)}, batch_size=[3])},
    ],
    ids=["non-tensor", "tuple-key", "device-less-child"],
)
def test_a_source_that_does_not_fit_goes_through_the_constructor(source):
    assert_same_state(
        build_tensordict(dict(source), [3], "cpu"), TensorDict(dict(source), batch_size=[3], device="cpu")
    )


def test_a_batch_mismatch_still_raises():
    with pytest.raises(RuntimeError, match="batch dimension mismatch"):
        build_tensordict({"a": torch.zeros(2)}, [3], "cpu")
