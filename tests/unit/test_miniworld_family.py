"""MiniWorldMovingBlockWorld State, restore, validation and rendering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.envs.miniworld.simulation.adapter import configure_pyglet

configure_pyglet()  # before miniworld imports pyglet.gl
pytest.importorskip("miniworld.envs.moving_blockworld")

import stable_envs as snv  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

pytestmark = pytest.mark.unit

SEED = 3
SIZE = 32


def _make(num_envs=2, **options):
    return snv.MiniWorldMovingBlockWorld(num_envs=num_envs, size=SIZE, device="cpu", **options)


def _actions(seed, steps, lanes=2):
    return np.random.default_rng(seed).integers(0, 9, size=(steps, lanes, 1))


@pytest.mark.parametrize("preset", ["flowm", "flowm-textured"])
def test_a_state_restores_into_an_env_with_another_world_and_history(preset):
    with _make(preset=preset) as source, _make(preset=preset) as restored:
        source.reset(seed=SEED)
        for action in _actions(0, 5):
            source.step(action, frames=None)
        saved = source.get_state()
        restored.reset(seed=SEED + 100)  # other rooms, block colours and textures
        for action in _actions(1, 3):
            restored.step(action, frames=None)
        assert not torch.equal(restored.get_state()["blocks", "color"], saved["blocks", "color"])
        assert restored.set_state(saved) is None
        assert_same_state(restored.get_state(), saved)
        torch.testing.assert_close(restored.render(), source.render(), rtol=0, atol=0)
        for action in _actions(2, 6):
            assert_same_state(restored.step(action), source.step(action))
        # The generator is part of State: unseeded resets continue identically.
        assert_same_state(restored.reset(), source.reset())


def test_a_carried_block_is_restored():
    with _make(num_envs=1, preset="upstream", agent_center_start=True) as env:
        env.reset(seed=SEED)
        for _ in range(60):  # pick up a block once one is in reach, turning to find one
            state = env.step(np.array([[5]]), frames=None)["state"]
            if state["agent", "carrying"].item() >= 0:
                break
            env.step(np.array([[0]]), frames=None)
        index = state["agent", "carrying"].item()
        assert index >= 0, "no block came within reach"
        saved = env.get_state()
        expected = [env.step(np.array([[a]]), frames=None) for a in (2, 1, 2)]  # the block moves with the agent
        env.step(np.array([[6]]), frames=None)  # drop it
        assert env.get_state()["agent", "carrying"].item() == -1
        env.set_state(saved)
        assert env.get_state()["agent", "carrying"].item() == index
        for a, want in zip((2, 1, 2), expected):
            assert_same_state(env.step(np.array([[a]]), frames=None), want)


def test_render_of_a_state_matches_the_live_frames_and_leaves_the_lanes_alone():
    with _make(num_envs=3) as env:
        env.reset(seed=SEED)
        env.step(_actions(4, 1, lanes=3)[0], frames=None)
        state = env.get_state()
        live = env.render()
        torch.testing.assert_close(env.render(state), live, rtol=0, atol=0)
        torch.testing.assert_close(env.render(state[2:3]), live[2:3], rtol=0, atol=0)
        # A stacked batch with repeated and reordered lanes.
        stacked = torch.cat([state[1:2], state[0:1], state[1:2]])
        torch.testing.assert_close(env.render(stacked), live[[1, 0, 1]], rtol=0, atol=0)
        assert_same_state(env.get_state(), state)
        assert live.shape == (3, SIZE, SIZE, 3) and live.dtype == torch.uint8
        assert live.float().std() > 10  # a drawn scene, not a cleared buffer


def _bad(state, key, value):
    bad = state.clone()
    bad[key] = value
    return bad


def test_invalid_states_are_rejected_before_any_lane_changes():
    with _make() as env:
        env.reset(seed=SEED)
        good = env.get_state()
        before = env.get_state()
        kind = good["blocks", "kind"].clone()
        texture = good["blocks", "texture"].clone()
        kind[1, 0], texture[1, 0] = 1, 3  # a textured ball on the second lane
        cases = [
            _bad(good, ("blocks", "color"), torch.full_like(good["blocks", "color"], 6)),
            _bad(
                good, ("agent", "carrying"), good["blocks", "present"].sum(-1, keepdim=True)
            ),  # past the lane's blocks
            _bad(good, ("blocks", "pos"), good["blocks", "pos"].float()),
            _bad(good, ("agent", "dir"), torch.full_like(good["agent", "dir"], float("nan"))),
            _bad(good, ("rng", "episode"), torch.full_like(good["rng", "episode"], -1)),
            _bad(_bad(good, ("blocks", "kind"), kind), ("blocks", "texture"), texture),
        ]
        missing = good.clone()
        del missing["rng", "seed"]
        cases.append(missing)
        for bad in cases:
            with pytest.raises((TypeError, ValueError)):
                env.set_state(bad)
            assert_same_state(env.get_state(), before)
        with pytest.raises(ValueError):
            env.set_state(good[:1])


def test_actions_must_be_action_ids():
    with _make() as env:
        with pytest.raises(ValueError):
            env.step(np.array([[0.0], [1.0]]))
        with pytest.raises(ValueError):
            env.step(np.array([[0], [9]]))
        assert env.action_spec.high == 8


def test_options_are_checked_and_shape_state():
    with pytest.raises(TypeError, match="unknown MovingBlockWorld option"):
        _make(sizes=3)
    with pytest.raises(ValueError, match="preset"):
        _make(preset="flowm-v2")
    with _make(num_envs=1, num_blocks=3, blocks_static=True, box_and_ball=False) as env:
        state = env.reset(seed=SEED)
        assert state["blocks", "kind"].shape == (1, 3) and not state["blocks", "kind"].any()
        assert not state["blocks", "velocity"].any()
        after = env.step(np.array([[4]]), frames=None)["state"]
        assert torch.equal(after["blocks", "pos"], state["blocks", "pos"])


def test_close_is_idempotent_and_refuses_later_calls():
    env = _make(num_envs=1)
    env.close()
    env.close()
    with pytest.raises(RuntimeError):
        env.reset(seed=0)
    with pytest.raises(RuntimeError):
        env.render()


@pytest.mark.parametrize("preset", ["flowm", "flowm-textured", "upstream"])
def test_numba_and_torch_give_the_same_episodes(preset):
    """Both kernels draw the same resets from the counter-based generator and step them alike.

    Torch's cosine and sine may differ from libm's in the last bit, so continuous motion agrees to 1e-9.
    """
    actions = _actions(5, 40, lanes=6)
    with _make(6, preset=preset) as numba, _make(6, preset=preset, simulation_backend="torch") as torch_env:
        assert (numba.simulation_backend, numba.execution_mode) == ("numba", "NATIVE_NUMBA")
        assert (torch_env.simulation_backend, torch_env.execution_mode) == ("torch", "NATIVE_TORCH")
        pairs = [(numba.reset(seed=SEED), torch_env.reset(seed=SEED))]
        pairs += [(numba.step(a, frames=None)["state"], torch_env.step(a, frames=None)["state"]) for a in actions]
        pairs.append((numba.reset(), torch_env.reset()))
        for want, got in pairs:
            for key in want.keys(True, True):
                torch.testing.assert_close(
                    got[key], want[key], rtol=0, atol=1e-9 if want[key].is_floating_point() else 0
                )


def test_the_simulation_backend_follows_the_device_and_the_cutoff():
    from stable_envs.envs.miniworld.simulation.batched import miniworld_backend

    assert miniworld_backend("cpu", 4096) == "numba"
    assert miniworld_backend("cuda", 1) == "triton"
    assert miniworld_backend("cuda", 1, "auto") == "triton"  # the default cutoff is 1
    assert miniworld_backend("cuda", 63, "auto", 64) == "numba"
    assert miniworld_backend("cuda", 64, "auto", 64) == "triton"
    assert miniworld_backend("cpu", 4096, "auto") == "numba"
    assert miniworld_backend("cpu", 1, "torch") == "torch"
    assert miniworld_backend("cuda", 4096, "numba") == "numba"
    for bad in (("cpu", 1, "warp"), ("cpu", 1, "triton"), ("cuda", 1, "auto", 0)):
        with pytest.raises(ValueError):
            miniworld_backend(*bad)
    with _make(1, simulation_backend="torch") as env:
        assert env.simulation_backend == "torch"


def test_inactive_environments_hold_their_state():
    """A sequence step: an environment that truncates holds its endpoint while the others go on."""
    for backend in ("numba", "torch"):
        with _make(2, preset="upstream", max_episode_steps=3, simulation_backend=backend) as env:
            env.reset(seed=SEED)
            state = env.get_state()
            state["lifecycle", "elapsed_steps"][1] = 1  # the second environment truncates first
            env.set_state(state)
            out = env.step(np.full((2, 4, 1), 2), frames=None)
            assert out["valid"].tolist() == [[True, True, True, False], [True, True, False, False]]
            assert out["state"]["lifecycle", "elapsed_steps"].tolist() == [[3], [3]]


@pytest.mark.parametrize("k", [0, 7])
def test_restoring_the_state_of_step_k_replays_the_trajectory_bit_for_bit(k):
    """Save at step ``k``, step on, restore, and repeat the same actions: State, outputs and frames match."""
    actions = _actions(21, k + 12)
    with _make() as env:
        env.reset(seed=SEED)
        for a in actions[:k]:
            env.step(a, frames=None)
        saved = env.get_state()
        first = [env.step(a) for a in actions[k:]]
        env.set_state(saved)
        for a, want in zip(actions[k:], first, strict=True):
            assert_same_state(env.step(a), want)


def test_each_environment_draws_its_block_count_at_reset():
    """FloWM presets draw 6 to 10 blocks per environment and reset; State pads to 10 and marks its blocks first."""
    with _make(num_envs=8) as env:
        counts = set()
        for seed in range(0, 60, 10):  # 48 distinct lane seeds
            state = env.reset(seed=seed)
            present = state["blocks", "present"]
            n = present.sum(-1)
            assert present.shape == (8, 10) and ((n >= 6) & (n <= 10)).all()
            assert torch.equal(present, torch.arange(10) < n[:, None])  # a lane's blocks come first
            counts |= set(n.tolist())
        assert counts == set(range(6, 11))
        assert_same_state(env.reset(seed=50), state)  # the same seed draws the same counts
        frames = env.render(state)
        torch.testing.assert_close(env.render(), frames, rtol=0, atol=0)


def test_render_takes_size_and_out_and_refuses_what_it_cannot_draw():
    with _make() as env:
        state = env.reset(seed=SEED)
        big = env.render(state, size=2 * SIZE)
        assert big.shape == (2, 2 * SIZE, 2 * SIZE, 3) and big.dtype == torch.uint8
        torch.testing.assert_close(env.render(size=2 * SIZE), big, rtol=0, atol=0)
        out = torch.empty(2, SIZE, SIZE, 3, dtype=torch.uint8)
        assert env.render(state, out=out) is out
        torch.testing.assert_close(out, env.render(state), rtol=0, atol=0)
        with pytest.raises(ValueError):
            env.render(state, out=torch.empty(2, SIZE, SIZE, 3, dtype=torch.float32))
        with pytest.raises(ValueError, match="supersampling"):
            env.render(state, supersample=2)
        with pytest.raises(ValueError, match="Variation"):
            env.render(state, variation={"color": 1})


def test_lanes_that_share_a_room_compile_its_display_list_once(monkeypatch):
    from miniworld.miniworld import MiniWorldEnv

    with _make(num_envs=4) as env:  # flowm: every lane is the same plain room
        env.reset(seed=SEED)
        compiles = []
        compile_static = MiniWorldEnv._render_static
        monkeypatch.setattr(MiniWorldEnv, "_render_static", lambda self: compiles.append(1) or compile_static(self))
        env.render()
        env.render()
        assert len(compiles) <= 1


@pytest.mark.parametrize("length", [1, 3])
def test_step_states_are_owned_copies(length):
    """``states`` repeats each State along time, in storage of its own (one step copies two flat buffers)."""
    with _make() as env:
        env.reset(seed=SEED)
        out = env.step(_actions(6, length).transpose(1, 0, 2), frames=None)
        states, state = out["states"], out["state"]
        assert states.batch_size == torch.Size([2, length])
        assert_same_state(states[:, -1], state)
        held = {state[key].untyped_storage().data_ptr() for key in state.keys(True, True)}
        for key in states.keys(True, True):
            assert states[key].untyped_storage().data_ptr() not in held, key
        before = state.clone()
        states.apply_(lambda x: x.zero_())
        assert_same_state(state, before)
