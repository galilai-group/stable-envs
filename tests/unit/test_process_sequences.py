"""Action sequences that run in one simulator loop, in the workers or in this process, give the outputs
of the per-action loop."""

import numpy as np
import pytest
import torch

from stable_envs._core.stepping import step
from stable_envs._execution.process import ProcessAdapter
from stable_envs.specs import Spec
from tests.helpers.reference.semantics import assert_same_state
from tests.unit.test_process_adapter import Lanes


class SequenceLanes(Lanes):
    process_sequences = True


class SequenceProcess(ProcessAdapter):
    name = "Lanes"


class _Env:
    """The members of an environment that the step dispatcher reads."""

    has_dynamics = True
    device = torch.device("cpu")

    def __init__(self, adapter):
        self._simulator = adapter
        self.num_envs = adapter.num_envs

    def _check_open(self, what):
        pass

    @property
    def action_spec(self):
        return Spec((1,), dtype=torch.float64)

    def _step_pipeline(self):
        return self._simulator

    def _step(self, actions):
        return self._simulator.step(actions)

    def get_state(self):
        return self._simulator.get_state()


def _run(adapter, actions, states):
    env = _Env(adapter)
    adapter.reset(0)
    outputs = [step(env, a, states=states, frames=None) for a in actions]
    return outputs, adapter.get_state()


@pytest.mark.parametrize("processes", [1, 3])
@pytest.mark.parametrize("capacity", [None, 3])
@pytest.mark.parametrize("states", [None, slice(-1, None), slice(None)])
def test_worker_sequences_match_the_per_action_loop(processes, capacity, states):
    rng = np.random.default_rng(5)
    # Lanes end at different actions, and in the last call every lane ends at the first action.
    actions = [torch.from_numpy(rng.uniform(0, 1, (6, 8, 1))) for _ in range(3)] + [torch.full((6, 8, 1), 9.0)]
    sequenced = SequenceProcess(SequenceLanes, 6, seed=2, processes=processes)
    looped = SequenceProcess(SequenceLanes, 6, seed=2, processes=processes)
    try:
        assert sequenced.sequence_steps > 0
        if capacity is not None:
            sequenced._sequence_steps = capacity
        looped._sequence_steps = 0
        expected, expected_end = _run(looped, actions, states)
        actual, actual_end = _run(sequenced, actions, states)
        for e, a in zip(expected, actual, strict=True):
            assert_same_state(a, e)
            assert list(a.keys(True, True)) == list(e.keys(True, True))
        assert_same_state(actual_end, expected_end)
        assert not bool(expected[0]["valid"].all()) and not bool(expected[-1]["valid"][:, 1:].any())
    finally:
        sequenced.close()
        looped.close()


class _LocalEnv(_Env):
    """An environment that steps its local simulator in this process."""

    def __init__(self, adapter, sequences):
        super().__init__(adapter)
        self.sequences = sequences

    def _step_pipeline(self):
        return None

    def _advance_without_snapshot(self, actions, *, active=None):
        return self._simulator.step(actions, active=active, capture_state=False)

    def _sequence_simulator(self):
        return self._simulator if self.sequences else None


@pytest.mark.parametrize("states", [None, slice(-1, None)])
def test_local_sequences_match_the_per_action_loop(states):
    rng = np.random.default_rng(7)
    actions = [torch.from_numpy(rng.uniform(0, 1, (6, 8, 1))) for _ in range(3)] + [torch.full((6, 8, 1), 9.0)]
    results = []
    for sequences in (False, True):
        adapter = SequenceLanes(6, 0)
        env = _LocalEnv(adapter, sequences)
        outputs = [step(env, a, states=states, frames=None) for a in actions]
        results.append((outputs, adapter.get_state()))
    (expected, expected_end), (actual, actual_end) = results
    for e, a in zip(expected, actual, strict=True):
        assert_same_state(a, e)
        assert list(a.keys(True, True)) == list(e.keys(True, True))
    assert_same_state(actual_end, expected_end)
    assert not bool(expected[0]["valid"].all())


def test_a_rejected_action_takes_the_actions_before_it():
    actions = torch.ones((4, 5, 1)) * 0.1
    actions[:, 3] = float("nan")
    for sequences in (True, False):
        adapter = SequenceProcess(SequenceLanes, 4, processes=2)
        try:
            if not sequences:
                adapter._sequence_steps = 0
            with pytest.raises(ValueError, match="finite"):
                step(_Env(adapter), actions.double(), states=None, frames=None)
            assert adapter.get_state()["lifecycle", "steps"].tolist() == [3, 3, 3, 3]
        finally:
            adapter.close()


def test_local_adapters_without_sequences_keep_the_per_action_loop():
    adapter = SequenceProcess(Lanes, 4, processes=2)
    try:
        assert adapter.sequence_steps == 0
        assert not any(key[0] == "sequence" for key in adapter._specs)
    finally:
        adapter.close()
