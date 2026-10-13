"""DM Control adapter State: owned typed snapshots, lane RNG replay, input validation."""

import importlib
import inspect
import pkgutil
from functools import partial

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.envs.dmcontrol.simulation.aligned import CONFIG as ALIGNED
from stable_envs.envs.dmcontrol.simulation.aligned import DMCAlignedAdapter
from stable_envs.envs.dmcontrol.simulation.continuation import CONFIG as CONTINUATION
from stable_envs.envs.dmcontrol.simulation.continuation import DMCContinuationAdapter
from stable_envs.envs.dmcontrol.simulation.first_family import FirstFamilyAdapter, ReacherAdapter
from stable_envs.envs.dmcontrol.simulation.humanoid import DMCHumanoidAdapter

CASES = [(FirstFamilyAdapter, (domain,)) for domain in ("acrobot", "pendulum", "cartpole", "cartpole-balance")] + [
    (ReacherAdapter, ()),
    (partial(ReacherAdapter, task="easy"), ()),
    *[(DMCAlignedAdapter, (domain,)) for domain in ALIGNED],
    *[(DMCContinuationAdapter, (domain,)) for domain in CONTINUATION],
    (DMCHumanoidAdapter, ()),
]


def equal(left, right):
    assert left.batch_size == right.batch_size
    assert set(left.keys(True, True)) == set(right.keys(True, True))
    for key in left.keys(True, True):
        assert left[key].dtype == right[key].dtype
        assert left[key].device == right[key].device
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True, msg=str(key))


@pytest.mark.parametrize("factory,args", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owned_typed_state_across_steps_reset_and_rng_boundary(factory, args, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA gate runs on Oscar")
    adapter = factory(*args, 3, device=device, seed=17)
    try:
        initial = adapter.get_state()
        retained = initial.clone()
        actions = np.full((3, adapter.envs[0].action_spec().shape[0]), 0.25)
        adapter.step(actions)
        adapter.reset(seed=43)
        equal(initial, retained)
        adapter.set_state(initial)
        equal(initial, adapter.get_state())

        # Lane 1 crosses a reset boundary and consumes a cached Gaussian; the
        # other lanes continue from different lifecycle positions.
        boundary = initial.clone()
        boundary["rng", "has_gauss"][1] = True
        boundary["rng", "cached_gaussian"][1] = 0.125
        boundary["lifecycle", "reset_next_step"][1] = True
        boundary["lifecycle", "step_count"][2] = 3
        adapter.set_state(boundary)
        equal(boundary, adapter.get_state())
        expected = adapter.step(actions)
        adapter.set_state(boundary)
        actual = adapter.step(actions)
        equal(expected, actual)
        held = actual.clone()
        # Caller writes must not mutate live physics or a sibling snapshot.
        live = adapter.get_state()
        for value in live.values(True, True):
            value.fill_(0)
        equal(held["state"], adapter.get_state())
    finally:
        adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.step(actions, capture_state=False)


@pytest.mark.parametrize("factory,args", CASES)
def test_malformed_input_is_rejected_without_touching_state(factory, args):
    adapter = factory(*args, 2, device="cpu", seed=17)
    try:
        state = adapter.reset(seed=17)
        before = state.clone()
        width = adapter.envs[0].action_spec().shape[0]
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(TensorDict({"qpos": state["qpos"]}, [2]))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((2, width + 1)))
        with pytest.raises(ValueError):
            adapter.step(np.full((2, width), np.nan))
        equal(before, adapter.get_state())
    finally:
        adapter.close()


def test_simulation_modules_do_not_import_stable_worldmodel():
    package = importlib.import_module("stable_envs.envs.dmcontrol.simulation")
    for info in pkgutil.iter_modules(package.__path__):
        source = inspect.getsource(importlib.import_module(f"{package.__name__}.{info.name}"))
        assert "stable_worldmodel" not in source, info.name
