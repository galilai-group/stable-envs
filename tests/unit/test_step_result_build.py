"""The step result of the lean host path equals the general path's exactly, and owns its memory."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch

import stable_envs as snv

pytestmark = pytest.mark.unit

FAMILIES = [
    pytest.param("PushT", marks=pytest.mark.family("pusht")),
    pytest.param("DMCReacher", marks=pytest.mark.family("dmc")),
    pytest.param("TwoRoom", marks=pytest.mark.family("navigation")),
    pytest.param("OGBCube", marks=pytest.mark.family("ogbench")),
]
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
]
CALLS = [
    {"states": None, "frames": None},
    {"frames": None},
    {"states": slice(1, None), "frames": None},
    {"states": slice(-1, None), "frames": None, "state": False},
]


def _same(actual, expected, where=""):
    """Keys in order, batch sizes, devices, dtypes, shapes, CPU strides and values (NaN equal to NaN)."""
    assert list(actual.keys(True)) == list(expected.keys(True)), where
    assert actual.batch_size == expected.batch_size and actual.device == expected.device, where
    for key in actual.keys(True):
        a, b = actual[key], expected[key]
        if not torch.is_tensor(a):
            assert a.batch_size == b.batch_size and a.device == b.device, (where, key)
            continue
        assert (a.dtype, a.device, a.shape) == (b.dtype, b.device, b.shape), (where, key)
        if a.numel() > 1 and a.device.type == "cpu":
            assert a.stride() == b.stride(), (where, key)
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=lambda m, k=key: f"{where} {k}: {m}")


def _spans(td):
    """``(first byte, end byte)`` of every non-empty leaf of ``td``."""
    out = []
    for value in td.values(True, True):
        if value.numel():
            start = value.data_ptr()
            end = start + value.element_size() * (1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride())))
            out.append((start, end))
    return out


def _overlaps(a, b):
    return any(x0 < y1 and y0 < x1 for x0, x1 in _spans(a) for y0, y1 in _spans(b))


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("length", [1, 16])
def test_lean_result_matches_the_general_path(name, device, num_envs, length):
    fast = snv.make(name, num_envs=num_envs, device=device)
    general = snv.make(name, num_envs=num_envs, device=device)
    with fast, general:
        general.__dict__["_host_step"] = None
        fast.reset(seed=3)
        general.reset(seed=3)
        rng = np.random.default_rng(4)
        kept = []
        for i, call in enumerate(CALLS * 2):
            action = torch.stack([fast.action_spec.sample(num_envs, rng=rng) for _ in range(length)], 1)
            a = fast.step(action.to(device), **call)
            b = general.step(action.to(device), **call)
            _same(a, b, f"step {i} {call}")
            if "state" in a.keys() and "states" in a.keys():
                assert not _overlaps(a["state"], a["states"]), "states shares memory with state"
            kept.append((a, b.clone()))
        # A kept result is not changed by later steps.
        for i, (a, b) in enumerate(kept):
            _same(a, b, f"kept step {i}")
        _same(fast.get_state(), general.get_state(), "final State")
