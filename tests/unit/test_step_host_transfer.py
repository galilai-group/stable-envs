"""Step outputs of process pipelines on an accelerator: host bookkeeping and one batched transfer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from stable_envs._core.tensordicts import to_device_batched


def test_a_batched_transfer_keeps_every_leaf_and_shares_no_memory():
    td = TensorDict(
        {
            "a": torch.arange(5, dtype=torch.float64),
            "b": {"c": torch.tensor([[True, False, True]] * 5), "d": torch.zeros(5, 0)},
            "e": torch.arange(10, dtype=torch.int32).reshape(2, 5).t(),
            "f": torch.arange(5, dtype=torch.uint8),
        },
        [5],
    )
    moved = to_device_batched(td, "cpu")
    assert moved.batch_size == td.batch_size and list(moved.keys(True, True)) == list(td.keys(True, True))
    for key in td.keys(True, True):
        a, b = td[key], moved[key]
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if a.numel():
            assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
