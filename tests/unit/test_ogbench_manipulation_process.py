"""Independent Cube/Scene ownership, staged restore, metadata and public chunks."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as se
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter, _worker
from tests.helpers.reference.semantics import assert_same_state


def failing_restore_worker(*args):
    # Spawn-safe injection after real source construction, on the final shard.
    start = args[2]
    original = OGBManipulationAdapter._restore

    def restore(adapter, row):
        candidate = original(adapter, row)
        if start > 0 and row["physics", "time"][0] == 123:
            candidate.close()
            raise RuntimeError("injected late-shard compilation failure")
        return candidate

    with patch.object(OGBManipulationAdapter, "_restore", restore):
        _worker(*args)


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_process_state_rng_variations_and_owned_outputs(family):
    local = OGBManipulationAdapter(family, 3, seed=17)
    remote = OGBManipulationProcessAdapter(family, 3, seed=17, processes=2)
    try:
        assert len(remote._metadata.envs) == 1 and not hasattr(remote, "envs")
        initial = local.get_state()
        assert_same_state(initial, remote.get_state())
        initial["rng", "task", "has_gauss"][2] = 1
        initial["rng", "task", "cached_gaussian"][2] = 0.25
        initial["rng", "environment", "has_uint32"][2] = 1
        initial["rng", "environment", "uinteger"][2] = 2**32 - 1
        for adapter in (local, remote):
            adapter.set_state(initial)
        actions = np.random.default_rng(41).uniform(-0.3, 0.3, (3, 8, 5))
        held = remote.get_state()
        for t in range(8):
            mask = None if t == 0 else torch.tensor([True, False, True])
            action = actions[:, t].astype(np.float32 if t % 2 else np.float64)
            assert_same_state(local.step(action, active=mask), remote.step(action, active=mask))
        assert_same_state(held, initial)
        remote.submit_step(actions[:, 0], capture_state=False)
        with pytest.raises(RuntimeError, match="in flight"):
            remote.get_state()
        # Pure validation must not communicate with in-flight workers.
        remote.validate(initial)
        assert_same_state(local.step(actions[:, 0], capture_state=False), remote.receive_step())
        assert_same_state(local.get_state(), remote.get_state())
        values = [{"cube.size": np.array([0.025]), "agent.color": np.array([0.1, 0.7, 0.4])}] * 3
        layout = local.make_variation_space().flat_layout()
        for adapter in (local, remote):
            adapter.configure(values)
        torch.testing.assert_close(
            local.variation_values(layout, anchor=True), remote.variation_values(layout, anchor=True), rtol=0, atol=0
        )
        assert_same_state(local.reset(), remote.reset())
        assert_same_state(
            local.reset(42, {"task_id": 4, "variation": []}), remote.reset(42, {"task_id": 4, "variation": []})
        )
        lanes = {"variation_values": {"cube.size": np.array([[0.012], [0.02], [0.028]])}}
        assert_same_state(local.reset(5, lanes), remote.reset(5, lanes))
        remote.set_state(initial)
        assert_same_state(initial, remote.get_state())
        snapshot = remote.get_state()
        for call in (
            lambda: remote.reset(-1),
            lambda: remote.reset(1, {"task_id": 99}),
            lambda: remote.configure([{}, {}, {"cube.size": [99]}]),
            lambda: remote.step(np.zeros((3, 5), np.int64)),
        ):
            with pytest.raises(ValueError):
                call()
            assert_same_state(snapshot, remote.get_state())
    finally:
        local.close()
        remote.close()
    assert remote._metadata._closed and all(not p.is_alive() for p in remote._processes)


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_all_shards_discard_after_real_late_restore_failure(family, monkeypatch):
    import stable_envs.envs.ogbench.simulation.manipulation_process as module

    monkeypatch.setattr(module, "_worker", failing_restore_worker)
    remote = OGBManipulationProcessAdapter(family, 2, seed=17, processes=2)
    try:
        before = remote.get_state()
        requested = before.clone()
        requested["physics", "time"][:] = 123
        with pytest.raises(RuntimeError, match="late-shard compilation"):
            remote.set_state(requested)
        assert not remote._closed and remote._pending is None
        assert_same_state(before, remote.get_state())
        # Both live owners are usable, and a subsequent valid transaction commits.
        remote.step(np.zeros((2, 5)))
        remote.set_state(before)
        assert_same_state(before, remote.get_state())
    finally:
        remote.close()


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_public_metadata_chunks_and_anchor_reset(family, monkeypatch):
    import stable_envs.envs.ogbench.simulation.manipulation as module

    with se.make("OGB" + family, num_envs=2, device="cpu", seed=17) as local:
        with monkeypatch.context() as m:
            m.setattr(module, "OGBManipulationAdapter", partial(OGBManipulationProcessAdapter, processes=2))
            remote = se.make("OGB" + family, num_envs=2, device="cpu", seed=17)
        with remote:
            before = local.get_state()
            for horizon in (1, 8):
                for selection in (None, slice(None), slice(None, None, -2), slice(9, 9)):
                    for env in (local, remote):
                        env.set_state(before)
                    actions = np.zeros((2, 5) if horizon == 1 else (2, horizon, 5), np.float64)
                    assert_same_state(
                        local.step(actions, states=selection, frames=None),
                        remote.step(actions, states=selection, frames=None),
                    )
            kwargs = {"cube.size": [0.025], "agent.color": [0.2, 0.7, 0.4]}
            torch.testing.assert_close(local.set_variation(kwargs), remote.set_variation(kwargs), rtol=0, atol=0)
            assert_same_state(local.reset(31, {"variation": []}), remote.reset(31, {"variation": []}))
            torch.testing.assert_close(local.get_style(), remote.get_style(), rtol=0, atol=0)
            for env in (local, remote):
                env.set_state(before)
            assert_same_state(local.to_render_state(), remote.to_render_state())


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_worker_death_closes_metadata_and_other_owners(family):
    remote = OGBManipulationProcessAdapter(family, 2, processes=2)
    remote._processes[0].terminate()
    remote._processes[0].join()
    with pytest.raises((RuntimeError, EOFError, BrokenPipeError, ConnectionError)):
        remote.get_state()
    assert remote._closed and remote._metadata._closed
    assert all(not p.is_alive() for p in remote._processes)
