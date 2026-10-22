"""The generic process adapter matches its local adapter, restores atomically and routes declared calls."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._execution.process import BROADCAST, PER_LANE, ProcessAdapter
from stable_envs._execution.protocol import SimulatorAdapter
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local
from tests.helpers.reference.semantics import assert_same_state


class Lanes(SimulatorAdapter):
    """Lanes that add a scaled action to a position and count their own steps."""

    process_methods = {"scale": PER_LANE, "total": BROADCAST, "offset": BROADCAST}
    process_attributes = ("action_dim",)
    action_dim = 1

    def __init__(self, count, seed=0):
        self.num_envs = count
        self.position = np.arange(count, dtype=np.float64) + (seed or 0)
        self.steps = np.zeros(count, np.int64)
        self.gain = np.ones(count)

    def get_state(self):
        return TensorDict(
            {"position": torch.tensor(self.position[:, None]), ("lifecycle", "steps"): torch.tensor(self.steps)},
            [self.num_envs],
            device="cpu",
        )

    def validate(self, state):
        if (state["position"] < 0).any():
            raise ValueError("Negative position")

    def set_state(self, state):
        self.validate(state)
        self.position = state["position"][:, 0].numpy().copy()
        self.steps = state["lifecycle", "steps"].numpy().copy()

    def reset(self, seed=None):
        self.position = np.arange(self.num_envs, dtype=np.float64) + (seed or 0)
        self.steps[:] = 0
        return self.get_state()

    def check_step(self, actions, active):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 1) or not np.isfinite(actions).all():
            raise ValueError("Lanes action must have finite shape (num_envs, 1)")
        return actions, None if active is None else np.asarray(active, bool)

    def step(self, actions, *, active=None, capture_state=True):
        lanes = np.ones(self.num_envs, bool) if active is None else np.asarray(active, bool)
        self.position[lanes] += self.gain[lanes] * np.asarray(actions)[lanes, 0]
        self.steps[lanes] += 1
        out = {"reward": torch.tensor(self.position.copy()), "terminated": torch.tensor(self.position > 4)}
        if capture_state:
            out["state"] = self.get_state()
        return TensorDict(out, [self.num_envs], device="cpu")

    def scale(self, values, offset=0.0):
        self.gain = np.asarray(values, np.float64) + offset

    def total(self):
        return np.array([self.position.sum()])

    def offset(self, value):
        return [value] * self.num_envs

    def close(self):
        pass


class LanesProcess(ProcessAdapter):
    name = "Lanes"


def _actions(count=8, batch=5):
    rng = np.random.default_rng(3)
    actions = [rng.uniform(0, 1, (batch, 1)) for _ in range(count)]
    masks = [None] + [torch.from_numpy(rng.random(batch) < 0.6) for _ in range(count - 1)]
    return actions, masks


def test_outputs_match_local_for_every_worker_count():
    actions, masks = _actions()
    assert_process_matches_local(
        lambda: Lanes(5, 3),
        lambda count: LanesProcess(Lanes, 5, seed=3, processes=count),
        actions,
        masks=masks,
        counts=(1, 2, 5),
        foreign=Lanes(5, 11).get_state(),
    )


def test_rejected_lane_restores_no_lane():
    adapter = LanesProcess(Lanes, 5, seed=3, processes=2)
    try:

        def corrupt(state):
            state["position"][0] += 1  # The first worker would accept its rows.
            state["position"][-1] = -1
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(np.ones((5, 1))))
        with pytest.raises(ValueError, match="Negative position"):
            adapter.set_state(corrupt(adapter.get_state()))
        bad = adapter.get_state()
        bad["position"][2] = float("nan")
        with pytest.raises(ValueError, match="Invalid State field"):
            adapter.set_state(bad)
        with pytest.raises(TypeError, match="Lanes requires full TensorDict State"):
            adapter.set_state(adapter.get_state()[:2])
        assert not adapter._closed and adapter._pending is None
    finally:
        adapter.close()


def test_declared_methods_split_lanes_and_join_results():
    adapter = LanesProcess(Lanes, 5, seed=0, processes=2)
    local = Lanes(5)
    try:
        for owner in (local, adapter):
            owner.scale(np.arange(5.0), 0.5)
            owner.step(np.ones((5, 1)))
        assert_same_state(adapter.get_state(), local.get_state())
        np.testing.assert_array_equal(adapter.total(), [local.position[:3].sum(), local.position[3:].sum()])
        assert adapter.offset(7) == [7] * 5
        with pytest.raises(ValueError, match="not a declared process method"):
            adapter.call("reset")
        with pytest.raises(AttributeError):
            adapter.missing  # noqa: B018
        assert not hasattr(adapter, "position")
    finally:
        adapter.close()


def test_capture_state_skips_the_snapshot_and_requests_are_exclusive():
    adapter = LanesProcess(Lanes, 4, processes=2)
    try:
        adapter.submit_step(np.ones((4, 1)), capture_state=False)
        with pytest.raises(RuntimeError, match="in flight"):
            adapter.get_state()
        result = adapter.receive_step()
        assert "state" not in result.keys()
        held = result.clone()
        adapter.step(np.ones((4, 1)))
        assert_same_state(result, held)
        with pytest.raises(RuntimeError, match="No Lanes step in flight"):
            adapter.receive_step()
        with pytest.raises(ValueError, match="action must have finite shape"):
            adapter.step(np.ones((3, 1)))
        assert_same_state(adapter.reset(10), Lanes(4, 10).get_state())
    finally:
        adapter.close()
    assert all(not process.is_alive() for process in adapter._processes)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()


def test_worker_death_closes_every_worker():
    adapter = LanesProcess(Lanes, 4, processes=2)
    adapter._processes[0].terminate()
    adapter._processes[0].join()
    with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
        adapter.get_state()
    assert adapter._closed and all(not process.is_alive() for process in adapter._processes)
