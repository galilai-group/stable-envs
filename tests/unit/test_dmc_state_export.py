"""DM Control adapter State: owned typed snapshots, lane RNG replay, input validation."""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._core.state_schema import get_field, schema_of
from stable_envs.envs.dmcontrol.simulation.aligned import CONFIG as ALIGNED
from stable_envs.envs.dmcontrol.simulation.aligned import DMCAlignedAdapter
from stable_envs.envs.dmcontrol.simulation.continuation import CONFIG as CONTINUATION
from stable_envs.envs.dmcontrol.simulation.continuation import DMCContinuationAdapter
from stable_envs.envs.dmcontrol.simulation.factory import FIELDS
from stable_envs.envs.dmcontrol.simulation.first_family import FirstFamilyAdapter, ReacherAdapter
from stable_envs.envs.dmcontrol.simulation.humanoid import DMCHumanoidAdapter
from stable_envs.envs.dmcontrol.simulation.state import field_name, state_schema, state_section
from tests.helpers.reference.semantics import assert_same_state
from tests.unit.test_state_schema import ADOPTED

CASES = [(FirstFamilyAdapter, (domain,)) for domain in ("acrobot", "pendulum", "cartpole", "cartpole-balance")] + [
    (ReacherAdapter, ()),
    *[(DMCAlignedAdapter, (domain,)) for domain in ALIGNED],
    *[(DMCContinuationAdapter, (domain,)) for domain in CONTINUATION],
    (DMCHumanoidAdapter, ()),
]


@pytest.mark.parametrize("factory,args", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owned_typed_state_across_steps_reset_and_rng_boundary(factory, args, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    adapter = factory(*args, 3, device=device, seed=17)
    try:
        initial = adapter.get_state()
        retained = initial.clone()
        actions = np.full((3, adapter.envs[0].action_spec().shape[0]), 0.25)
        adapter.step(actions)
        adapter.reset(seed=43)
        assert_same_state(initial, retained)
        adapter.set_state(initial)
        assert_same_state(initial, adapter.get_state())

        # Lane 1 crosses a reset boundary and consumes a cached Gaussian; the
        # other lanes continue from different lifecycle positions.
        boundary = initial.clone()
        get_field(boundary, "rng/has_gauss")[1] = True
        get_field(boundary, "rng/cached_gaussian")[1] = 0.125
        get_field(boundary, "lifecycle/reset_next_step")[1] = True
        get_field(boundary, "lifecycle/step_count")[2] = 3
        adapter.set_state(boundary)
        assert_same_state(boundary, adapter.get_state())
        expected = adapter.step(actions)
        adapter.set_state(boundary)
        actual = adapter.step(actions)
        assert_same_state(expected, actual)
        held = actual.clone()
        # Caller writes must not mutate live physics or a sibling snapshot.
        live = adapter.get_state()
        for value in live.values(True, True):
            value.fill_(0)
        assert_same_state(held["state"], adapter.get_state())
    finally:
        adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.step(actions, capture_state=False)


def test_each_simulator_folds_its_fields_into_three_groups_of_its_own():
    assert {state_section(key) for key in FIELDS} == set(ADOPTED["dmcontrol"][0])
    groups = set()
    for key, fields in FIELDS.items():
        schema = state_schema(key, fields)
        assert [group.path for group in schema.groups.values()] == [
            (state_section(key), dtype) for dtype in ("float64", "int64", "bool")
        ]
        assert set(schema.names) == {field_name(name) for name in fields}
        assert not groups & set(schema.groups)
        groups |= set(schema.groups)


@pytest.mark.parametrize("factory,args", CASES)
def test_exported_state_is_folded_and_reads_back_by_field(factory, args):
    adapter = factory(*args, 2, device="cpu", seed=17)
    try:
        state = adapter.reset(seed=17)
        assert schema_of(state) == adapter.schema
        assert len(list(state.keys(True, True))) == 3
        adapter.schema.validate(state, finite=True)
        assert torch.equal(get_field(state, "lifecycle/step_count"), torch.zeros(2, 1, dtype=torch.int64))
        assert get_field(state, "rng/keys").shape == (2, 624)
    finally:
        adapter.close()


@pytest.mark.parametrize("factory,args", CASES)
def test_malformed_input_is_rejected_without_touching_state(factory, args):
    adapter = factory(*args, 2, device="cpu", seed=17)
    try:
        state = adapter.reset(seed=17)
        before = state.clone()
        width = adapter.envs[0].action_spec().shape[0]
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(TensorDict({"qpos": get_field(state, "qpos")}, [2]))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((2, width + 1)))
        with pytest.raises(ValueError):
            adapter.step(np.full((2, width), np.nan))
        assert_same_state(before, adapter.get_state())
    finally:
        adapter.close()


def test_simulation_modules_do_not_import_stable_worldmodel():
    package = importlib.import_module("stable_envs.envs.dmcontrol.simulation")
    for info in pkgutil.iter_modules(package.__path__):
        source = inspect.getsource(importlib.import_module(f"{package.__name__}.{info.name}"))
        assert "stable_worldmodel" not in source, info.name
