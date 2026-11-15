"""A step of one action without frames on a host-step family takes the short path with the same result."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch

import stable_envs as snv
from stable_envs._core import stepping
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

FAMILIES = [
    pytest.param("DMCReacher", {}, marks=pytest.mark.family("dmc")),
    pytest.param("DMCCartpoleSwingup", {}, marks=pytest.mark.family("dmc")),
    pytest.param("TwoRoom", {"simulation_backend": "numba"}, marks=pytest.mark.family("navigation")),
]
CALLS = [
    {"states": None, "frames": None},
    {"frames": None},
    {"states": slice(1, None), "frames": None},
    {"frames": None, "state": False},
    {"states": None, "frames": None, "state": False},
]


def _same(actual, expected):
    assert list(actual.keys(True)) == list(expected.keys(True))
    assert_same_state(actual, expected)
    for key in actual.keys(True):
        node = actual[key]
        if not torch.is_tensor(node):
            assert node.batch_size == expected[key].batch_size, key


@pytest.mark.parametrize("name,kwargs", FAMILIES)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("sequence", [False, True])
def test_short_path_matches_the_general_path(name, kwargs, num_envs, sequence, monkeypatch):
    calls = []
    short = stepping._host_single_step

    def counted(env, *args):
        out = short(env, *args)
        if env is fast:
            calls.append(out is not None)
        return out

    monkeypatch.setattr(stepping, "_host_single_step", counted)
    fast = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    general = snv.make(name, num_envs=num_envs, device="cpu", **kwargs)
    with fast, general:
        general.__dict__["_host_step"] = None
        fast.reset(seed=5)
        general.reset(seed=5)
        rng = np.random.default_rng(2)
        previous = None
        for call in CALLS * 2:
            action = fast.action_spec.sample(num_envs, rng=rng)
            if sequence:
                action = action.unsqueeze(1)
            a, b = fast.step(action, **call), general.step(action, **call)
            _same(a, b)
            if previous is not None:
                # A result owns its memory: later steps don't change it.
                _same(previous[0], previous[1])
            previous = (a, b.clone())
        _same(fast.get_state(), general.get_state())
    assert calls and all(calls)
