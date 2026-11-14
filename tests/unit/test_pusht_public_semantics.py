"""Public checkpoint/branch contract, independently for both PushT action spaces."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state


@pytest.fixture(params=["PushT", "PushTDiscrete"])
def name(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def actions(name, batch, length=1):
    rng = np.random.default_rng(54)
    return (
        rng.integers(0, 16, (batch, length, 2), dtype=np.int64)
        if name == "PushTDiscrete"
        else rng.uniform(-1, 1, (batch, length, 2)).astype(np.float32)
    )


def test_disk_checkpoint_cross_instance_continuation(name, device, tmp_path):
    with (
        se.make(name, num_envs=2, size=64, device=device) as original,
        se.make(name, num_envs=2, size=64, device=device, seed=999) as restored,
    ):
        original.step(actions(name, 2, 8), frames=None)
        state = original.get_state()
        path = tmp_path / "state.npz"
        np.savez_compressed(path, **{"/".join(k): v.cpu().numpy() for k, v in state.items(True, True)})
        with np.load(path, allow_pickle=False) as data:
            loaded = TensorDict(
                {tuple(k.split("/")): torch.tensor(data[k], device=device) for k in data}, [2], device=device
            )
        restored.set_state(loaded)
        assert_same_state(state, restored.get_state())
        assert restored.state_spec.contains(restored.get_state())
        assert torch.equal(original.render(state), restored.render())
        future = actions(name, 2, 8)
        assert_same_state(original.step(future, frames=slice(None)), restored.step(future, frames=slice(None)))
        assert_same_state(state, loaded)


@pytest.mark.parametrize(
    "selection", [None, slice(None), slice(-1, None), slice(None, None, -1), slice(None, None, 2), slice(20, 30)]
)
def test_public_chunk_slices_restore_and_hold_owned_snapshots(name, device, selection):
    with se.make(name, num_envs=2, size=32, device=device) as env:
        start = env.get_state()
        a = actions(name, 2, 4)
        expected = env.step(a, frames=None)["state"]
        env.set_state(start)
        out = env.step(a, states=selection, frames=selection)
        assert_same_state(expected, out["state"])
        saved = out.clone()
        if selection is not None:
            for index in range(out["states"].batch_size[1]):
                selected = out["states"][:, index]
                env.set_state(selected)
                assert torch.equal(env.render(), out["frames"][:, index])
        env.step(a[:, 0], frames=None)
        assert_same_state(out, saved)


def test_failed_restore_and_bad_actions_do_not_partially_mutate_batch(name, device):
    with se.make(name, num_envs=2, size=32, device=device) as env:
        before = env.step(actions(name, 2, 3), frames=None)["state"]
        bad = before.clone()
        bad["current", "body", "block", "position"][1, 0] += 1
        with pytest.raises(ValueError, match="replay disagrees"):
            env.set_state(bad)
        assert_same_state(before, env.get_state())
        a = actions(name, 2)[:, 0]
        a[1, 0] = 1000 if name == "PushTDiscrete" else np.nan
        with pytest.raises(ValueError):
            env.step(a, frames=None)
        assert_same_state(before, env.get_state())


def test_render_projection_has_no_history_and_does_not_step(name, monkeypatch):
    with se.make(name, size=96, device="cpu", render_action=True) as env:
        state = env.step(actions(name, 1, 8), frames=None)["state"]
        first = env.render(state)
        projected = env.to_render_state(state)
        assert "history" not in projected.keys()
        assert ("current", "goal_image") not in projected.keys(True, True)

        def forbidden(*args, **kwargs):
            raise AssertionError("Rendering must not step or replay physics")

        for replay in env._simulator.envs:
            monkeypatch.setattr(replay.env, "step", forbidden)
        for cached in env._projection.scenes:
            monkeypatch.setattr(cached[1], "step", forbidden)
        assert torch.equal(first, env.render(projected))
        assert_same_state(state, env.get_state())
        assert ("current", "body", "agent", "velocity") not in projected.keys(True, True)
        undrawn = state.clone()
        undrawn["current", "body", "agent", "velocity"][0, 0] = float("nan")
        assert torch.equal(first, env.render(undrawn))
        drawn = state.clone()
        drawn["current", "body", "agent", "position"][0, 0] = float("nan")
        with pytest.raises(ValueError, match="Nonfinite PushT render field"):
            env.render(drawn)


def test_persistent_variation_and_reset_checkpoint(name):
    with se.make(name, size=32, device="cpu") as env:
        env.set_variation({"agent.shape": 3, "block.shape": 7, "agent.scale": 25.0})
        state = env.get_state()
        expected = env.reset(seed=53, options={"variation": []})
        env.set_state(state)
        assert_same_state(expected, env.reset(seed=53, options={"variation": []}))
        before = env.get_state()
        with pytest.raises(ValueError):
            env.set_variation({"agent.shape": 4, "block.shape": 999})
        assert_same_state(before, env.get_state())


def test_set_variation_applies_appearance_now_and_bodies_at_reset(name):
    with se.make(name, num_envs=2, size=32, device="cpu") as env:
        env.reset(seed=8, options={"variation": [], "variation_values": {"block.shape": np.asarray([1, 4])}})
        moment = env.get_state()["current", "body", "block", "moment"].clone()
        env.set_variation({"agent.color": np.asarray([[1, 2, 3], [4, 5, 6]], np.uint8), "block.shape": 6})
        col = {p: o for p, o, _ in env.variation_space.flat_layout()}
        assert env.get_variation()[:, col["agent.color"]].tolist() == [1.0, 4.0]
        assert env.get_variation()[:, col["block.shape"]].tolist() == [1.0, 4.0]
        assert torch.equal(env.get_state()["current", "body", "block", "moment"], moment)
        env.reset(seed=8, options={"variation": []})
        assert env.get_variation()[:, col["block.shape"]].tolist() == [6.0, 6.0]


def test_public_device_and_backend_are_explicit(name):
    with se.make(name, device="cpu", size=32) as env:
        assert env.backend == "pygame"
        assert env.action_spec.shape == (2,)
        assert env.action_spec.discrete_n == (16 if name == "PushTDiscrete" else None)
        assert env.action_spec.contains(torch.tensor(actions(name, 1)[:, 0]))
        state = env.get_state()
    for call in (env.get_state, lambda: env.set_state(state), env.render, env.reset):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    with pytest.raises(ValueError, match="pygame"):
        se.make(name, backend="cuda")


def test_flat_sampled_variation_is_valid_and_restorable(name):
    with se.make(name, num_envs=2, size=32, device="cpu") as env:
        env.set_variation(env.sample_variation())
        state = env.reset(seed=27, options={"variation": []})
        env.step(actions(name, 2)[:, 0], frames=None)
        env.set_state(state)
        assert_same_state(state, env.get_state())


def test_render_names_what_pusht_accepts_and_construction_does_not_warn(recwarn):
    with se.make("PushT", num_envs=2, size=64, device="cpu") as env:
        state = env.reset(seed=0)
        with pytest.raises(TypeError, match="keeps its variation in the State"):
            env.render(state, variation=env.sample_variation(2))
        with pytest.raises(TypeError, match="accepts size= and out="):
            env.render(state, supersample=2)
    assert not [w for w in recwarn if "Casting input x" in str(w.message)]
