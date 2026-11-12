"""PushT process lanes match local lanes, through the adapter and the public env."""

import numpy as np
import pytest
import torch

pytest.importorskip("pymunk")

import stable_envs as se  # noqa: E402
from stable_envs.envs.pusht.simulation import execution  # noqa: E402
from stable_envs.envs.pusht.simulation.numba_batch import PushTNumbaAdapter  # noqa: E402
from stable_envs.envs.pusht.simulation.process import PushTProcessAdapter  # noqa: E402
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

CONFIG = {"resolution": 32}


@pytest.mark.parametrize("discrete", [False, True])
def test_process_matches_local(discrete):
    rng = np.random.default_rng(5)
    if discrete:
        actions = list(rng.integers(0, 16, (10, 5, 2)).astype(np.int64))
    else:
        actions = [a.astype(np.float32 if t % 2 else np.float64) for t, a in enumerate(rng.uniform(-1, 1, (10, 5, 2)))]
    masks = [None] + [torch.from_numpy(rng.random(5) < 0.6) for _ in range(9)]
    other = PushTReplayAdapter(5, seed=99, discrete=discrete, config=CONFIG)
    for action in actions[:3]:
        other.step(action, active=torch.tensor([True, False, True, True, False]))
    foreign = other.get_state()
    other.close()
    assert_process_matches_local(
        lambda: PushTReplayAdapter(5, seed=17, discrete=discrete, config=CONFIG),
        lambda count: PushTProcessAdapter(5, seed=17, discrete=discrete, config=CONFIG, processes=count),
        actions,
        masks=masks,
        counts=(1, 2, 5),
        foreign=foreign,
    )


def test_invalid_last_lane_restores_no_lane():
    adapter = PushTProcessAdapter(4, seed=17, config=CONFIG, processes=2)
    try:
        adapter.step(np.full((4, 2), 0.5), active=np.array([True, False, True, True]))

        def corrupt(state):
            state["current", "body", "block", "position"][0, 0] += 1
            state["recipe", "version"][-1] += 1
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(np.zeros((4, 2))))
        with pytest.raises(ValueError, match="PushT action must be finite"):
            adapter.step(np.full((4, 2), np.nan))
        with pytest.raises(ValueError, match="Unsupported PushT reset option"):
            adapter.reset(3, {"unknown": 1})
        assert not adapter._closed
    finally:
        adapter.close()


def test_public_env_with_workers_matches_local(monkeypatch):
    local = se.make("PushT", num_envs=4, size=32, device="cpu", seed=17)
    with monkeypatch.context() as patch:
        patch.setattr(execution, "process_count", lambda num_envs, cpus, **_: 2)
        remote = se.make("PushT", num_envs=4, size=32, device="cpu", seed=17)
    try:
        # Workers reset the lanes; the batched Numba step runs in this process.
        assert isinstance(remote._simulator, PushTNumbaAdapter) and remote._simulator._pool is not None
        pool = remote._simulator._pool
        actions = np.random.default_rng(3).uniform(-1, 1, (4, 6, 2)).astype(np.float32)
        for env in (local, remote):
            env.set_variation({"agent.color": np.arange(12, dtype=np.uint8).reshape(4, 3), "agent.scale": 25.0})
        torch.testing.assert_close(local.get_variation(), remote.get_variation(), rtol=0, atol=0)
        assert_same_state(local.reset(seed=5), remote.reset(seed=5))
        for states, frames in ((slice(None), slice(-1, None)), (None, None), (slice(None, None, 2), slice(None))):
            expected = local.step(actions, states=states, frames=frames)
            actual = remote.step(actions, states=states, frames=frames)
            assert_same_state(actual, expected)
        saved = local.get_state()
        for env in (local, remote):
            env.set_state(saved)
        torch.testing.assert_close(local.render(), remote.render(), rtol=0, atol=0)
        rows = local.sample_variation(rng=np.random.default_rng(1))
        torch.testing.assert_close(local.set_variation(rows), remote.set_variation(rows), rtol=0, atol=0)
        lanes = {"variation": [], "variation_values": {"block.shape": np.asarray([1, 4, 2, 6])}}
        assert_same_state(local.reset(seed=8, options=lanes), remote.reset(seed=8, options=lanes))
        assert_same_state(local.step(actions[:, 0]), remote.step(actions[:, 0]))
    finally:
        local.close()
        remote.close()
    assert all(not p.is_alive() for p in pool._processes)


def test_routing_keeps_small_batches_local():
    with se.make("PushT", num_envs=2, size=32, device="cpu") as env:
        assert isinstance(env._simulator, PushTReplayAdapter)
