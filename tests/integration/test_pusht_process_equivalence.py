"""Legacy Pymunk reference batching and public JAX routing."""

import numpy as np
import pytest
import torch

pytest.importorskip("pymunk")

import stable_envs as se  # noqa: E402
from stable_envs.envs.pusht.simulation.process import PushTProcessAdapter  # noqa: E402
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402

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
        with pytest.raises(ValueError, match="PushT continuous actions must be in"):
            adapter.step(np.full((4, 2), 2.0))
        with pytest.raises(ValueError, match="Unsupported PushT reset option"):
            adapter.reset(3, {"unknown": 1})
        assert not adapter._closed
    finally:
        adapter.close()


@pytest.mark.parametrize("batch", [1, 64])
def test_public_routing_uses_jax_without_pymunk_stepping(batch, monkeypatch):
    from stable_envs.envs.pusht.simulation.jax_batch import PushTJaxAdapter

    def forbidden(*args, **kwargs):
        raise AssertionError("Pymunk stepping in public PushT")

    with se.make("PushT", num_envs=batch, size=32, device="cpu") as env:
        assert isinstance(env._simulator, PushTJaxAdapter)
        for owner in env._simulator.envs:
            monkeypatch.setattr(owner.env.space, "step", forbidden)
        env.step(np.zeros((batch, 2), np.float32), frames=None)
