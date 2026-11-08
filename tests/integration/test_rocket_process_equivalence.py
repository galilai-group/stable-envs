"""Rocket process lanes match local lanes, through the adapter and the public env, and large batches split."""

import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")

import stable_envs as se  # noqa: E402
from stable_envs import tuning  # noqa: E402
from stable_envs._execution import policy, render_pool  # noqa: E402
from stable_envs.envs.rocket import legacy  # noqa: E402
from stable_envs.envs.rocket.simulation import execution  # noqa: E402
from stable_envs.envs.rocket.simulation.adapter import RocketAdapter  # noqa: E402
from stable_envs.envs.rocket.simulation.process import LANES_PER_PROCESS, RocketProcessAdapter  # noqa: E402
from tests.helpers.process_equivalence import assert_atomic_restore, assert_process_matches_local  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

LOW = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0])
HIGH = np.ones(7)


def _actions(batch, steps, seed=5):
    rng = np.random.default_rng(seed)
    actions = rng.uniform(LOW, HIGH, (steps, batch, 7))
    actions[:, 1::2, 3] = 0  # Odd lanes never ignite.
    actions[:, ::2, 3] = 1
    return [a.astype(np.float32 if t % 2 else np.float64) for t, a in enumerate(actions)]


def test_process_matches_local():
    rng = np.random.default_rng(7)
    masks = [None] + [torch.from_numpy(rng.random(5) < 0.6) for _ in range(9)]
    other = RocketAdapter(5, "cpu", 99)
    for action in _actions(5, 3, seed=9):
        other.step(action)
    foreign = other.get_state()
    other.close()
    assert_process_matches_local(
        lambda: RocketAdapter(5, "cpu", 17),
        lambda count: RocketProcessAdapter(5, "cpu", 17, processes=count),
        _actions(5, 10),
        masks=masks,
        counts=(1, 2, 5),
        foreign=foreign,
    )


def test_replay_mismatch_in_one_worker_restores_no_lane():
    adapter = RocketProcessAdapter(4, "cpu", 17, processes=2)
    try:
        for action in _actions(4, 3):
            adapter.step(action)

        def corrupt(state):
            state["current", "wrapper", "state"][0, 0] += 1e-3
            state["current", "wrapper", "state"][-1, 0] += 1e-3
            return state

        assert_atomic_restore(adapter, corrupt, ValueError, lambda a: a.step(_actions(4, 1)[0]))
        with pytest.raises(AssertionError, match="ignition"):
            bad = np.zeros((4, 7))
            bad[-1, 3] = 2
            adapter.step(bad)
        with pytest.raises(ValueError, match="seed"):
            adapter.reset(-1)
        with pytest.raises(ValueError, match="Unsupported Rocket reset option"):
            adapter.reset(3, {"unknown": True})
        assert not adapter._closed and adapter._pending is None
    finally:
        adapter.close()


def test_public_env_with_workers_matches_local(monkeypatch):
    with se.tuning.override(workers=0):
        local = se.make("PyFlytRocketLanding", num_envs=4, size=64, device="cpu", seed=17)
    with monkeypatch.context() as patch:
        patch.setattr(execution, "process_count", lambda num_envs, cpus, **_: 2)
        remote = se.make("PyFlytRocketLanding", num_envs=4, size=64, device="cpu", seed=17)
    try:
        assert isinstance(local._simulator, RocketAdapter)
        assert isinstance(remote._simulator, RocketProcessAdapter)
        actions = torch.as_tensor(np.stack(_actions(4, 8), axis=1))
        for env in (local, remote):
            env.set_variation({"rocket.body_color": np.array([10.0, 200.0, 30.0])})
        torch.testing.assert_close(local.get_variation(), remote.get_variation(), rtol=0, atol=0)
        values = {"variation_values": {"environment.start_height_ratio": np.linspace(0.75, 0.9, 4)}}
        assert_same_state(local.reset(seed=5, options=values), remote.reset(seed=5, options=values))
        for states, frames in ((slice(None), slice(-1, None)), (None, None), (slice(None, None, 3), slice(None))):
            assert_same_state(
                remote.step(actions, states=states, frames=frames), local.step(actions, states=states, frames=frames)
            )
        state = remote.get_state()
        ignited = torch.isclose(state["render", "rgba"][:, 5], torch.tensor(legacy.IGNITION_RGBA, dtype=torch.float64))
        assert ignited.all(-1).tolist() == [True, False, True, False]
        torch.testing.assert_close(local.render(), remote.render(), rtol=0, atol=0)
        # Live appearance for supplied pose rows keeps the flame off.
        style = remote.resolve_style().values
        torch.testing.assert_close(local.resolve_style().values, style, rtol=0, atol=0)
        assert not style[:, legacy.style_layout()["flame.ignited"][0]].any()
        saved = local.get_state()
        for env in (local, remote):
            env.set_state(saved)
        assert_same_state(local.step(actions[:, 0]), remote.step(actions[:, 0]))
    finally:
        local.close()
        remote.close()
    assert all(not p.is_alive() for p in remote._simulator._processes)


def test_large_batches_split_across_processes(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    assert policy.process_count(1025, None, lanes_per_process=LANES_PER_PROCESS) == 3
    assert policy.process_count(LANES_PER_PROCESS, None, lanes_per_process=LANES_PER_PROCESS) == 0
    assert policy.process_count(1025, None, eligible=False, lanes_per_process=LANES_PER_PROCESS) == 0
    with pytest.raises(ValueError, match="at most"):
        RocketProcessAdapter(LANES_PER_PROCESS + 1, "cpu", 0, processes=1)


@pytest.mark.slow
def test_a_batch_above_one_process_restores():
    batch = LANES_PER_PROCESS + 3
    with se.make("PyFlytRocketLanding", num_envs=batch, size=32, device="cpu", seed=3) as env:
        assert isinstance(env._simulator, RocketProcessAdapter) and len(env._simulator._processes) >= 2
        actions = torch.as_tensor(np.stack(_actions(batch, 2), axis=1))
        saved = env.get_state()
        first = env.step(actions, frames=None)
        env.set_state(saved)
        assert_same_state(env.step(actions, frames=None), first)


@pytest.mark.parametrize("workers", [0, 2])
def test_step_frames_match_state_render(monkeypatch, workers):
    with monkeypatch.context() as patch:
        patch.setattr(execution, "process_count", lambda num_envs, cpus, **_: workers)
        env = se.make("PyFlytRocketLanding", num_envs=4, size=40, supersample=2, device="cpu", seed=3)
    try:
        assert isinstance(env._simulator, RocketProcessAdapter) == bool(workers)
        env.set_variation({"rocket.body_color": np.array([10.0, 200.0, 30.0])})
        actions = torch.as_tensor(np.stack(_actions(4, 6), axis=1))
        env.reset(seed=11)
        out = env.step(actions, states=slice(None), frames=slice(None))
        for t in range(actions.shape[1]):
            torch.testing.assert_close(out["frames"][:, t], env.render(out["states"][:, t]), rtol=0, atol=0)
        # RenderStates of other batch sizes give the same frames.
        first, last = env.to_render_state(out["states"][:, 0]), env.to_render_state(out["states"][:, -1])
        torch.testing.assert_close(env.render(first[1:3]), out["frames"][1:3, 0], rtol=0, atol=0)
        frames = torch.cat([out["frames"][:, 0], out["frames"][:, -1]])
        torch.testing.assert_close(env.render(torch.cat([first, last])), frames, rtol=0, atol=0)
        env.reset(seed=11)
        frames = env.step(actions, states=None, frames=slice(None))["frames"]
        torch.testing.assert_close(frames, out["frames"], rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.skipif(
    sys.platform != "linux" or len(getattr(os, "sched_getaffinity", lambda _: ())(0)) < 2,
    reason="render workers run on Linux with at least 2 CPUs",
)
def test_state_render_in_render_workers_matches_in_process():
    with se.make("PyFlytRocketLanding", num_envs=5, size=40, supersample=2, device="cpu", seed=3) as env:
        env.set_variation({"rocket.body_color": np.array([10.0, 200.0, 30.0])})
        env.reset(seed=11)
        out = env.step(torch.as_tensor(np.stack(_actions(5, 8), axis=1)), states=slice(None), frames=None)
        state = out["states"][:, -1]
        rows = torch.cat([env.to_render_state(out["states"][:, t]) for t in (1, 4, 7)])
        with tuning.override(render_workers=0):
            serial = env.render(state), env.render(rows)
        with tuning.override(render_workers=3):
            split = env.render(state), env.render(rows)
            # The main process checks the State while the workers draw; a State that fails gives no frames.
            bad = state.clone()
            bad["history", "valid"][1, 0] = False
            with pytest.raises(ValueError, match="history mask"):
                env.render(bad)
            again = env.render(state)
        pools = render_pool._POOLS.items()
        assert any(key[0] == "rocket-tiny" and isinstance(pool, render_pool.RenderPool) for key, pool in pools)
        for a, b in zip(serial, split, strict=True):
            assert a.dtype == torch.uint8 and torch.equal(a, b)
        assert torch.equal(again, serial[0])
