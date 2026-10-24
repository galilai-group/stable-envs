"""MNISTWorld State, restore, validation, presets and rendering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs.envs.mnist_world import data
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

SEED = 5


def _make(num_envs=2, **options):
    return snv.MNISTWorld(num_envs=num_envs, device="cpu", **options)


def _actions(seed, steps, lanes=2, bound=10):
    return np.random.default_rng(seed).integers(-bound, bound + 1, size=(steps, lanes, 2))


def test_a_state_restores_into_an_env_with_another_world_and_history():
    with _make() as source, _make() as restored:
        source.reset(seed=SEED)
        for action in _actions(0, 5):
            source.step(action, frames=None)
        saved = source.get_state()
        restored.reset(seed=SEED + 100)
        for action in _actions(1, 3):
            restored.step(action, frames=None)
        assert not torch.equal(restored.get_state()["digits", "image"], saved["digits", "image"])
        assert restored.set_state(saved) is None
        assert_same_state(restored.get_state(), saved)
        torch.testing.assert_close(restored.render(), source.render(), rtol=0, atol=0)
        for action in _actions(2, 6):
            assert_same_state(restored.step(action), source.step(action))
        # The sampler is part of State: unseeded resets continue identically.
        assert_same_state(restored.reset(), source.reset())
        assert restored.get_state()["rng", "episode"].tolist() == [[1], [1]]


def test_lane_i_of_a_seeded_reset_does_not_depend_on_the_batch():
    with _make(num_envs=3) as wide, _make(num_envs=1) as narrow:
        state = wide.reset(seed=SEED)
        assert_same_state(narrow.reset(seed=SEED + 2), state[2:3])
        assert_same_state(narrow.reset(), wide.reset()[2:3])


def test_digits_move_by_their_velocity_and_the_camera_by_the_action():
    with _make(num_envs=1) as env:
        state = env.reset(seed=SEED)
        action = np.array([[7, -10]])
        after = env.step(action, frames=None)["state"]
        world = env.config["world_size"]
        moved = (state["digits", "position"] + state["digits", "velocity"]) % world
        assert torch.equal(after["digits", "position"], moved)
        assert after["camera"].tolist() == [[7, 40]]
        assert after["step"].tolist() == [[1]]


def test_the_view_is_the_wrapped_sum_of_the_digits_around_the_camera():
    images, labels = data.mnist_train(str(data.default_dir()))
    with _make(num_envs=1) as env:
        state = env.reset(seed=SEED)
        state["digits", "position"] = torch.tensor([[48, 45, 2, 3, 20, 20, 20, 30, 0, 0]])
        state["digits", "velocity"].zero_()
        state["camera"] = torch.tensor([[1, 47]])
        env.set_state(state)
        world, window = 50, 32
        canvas = np.zeros((world, world), np.int32)
        for k, image in enumerate(state["digits", "image"][0].tolist()):
            x, y = state["digits", "position"][0, 2 * k : 2 * k + 2].tolist()
            rows = (y + np.arange(28)) % world
            cols = (x + np.arange(28)) % world
            canvas[np.ix_(rows, cols)] += images[image]
        rows = (47 - window // 2 + np.arange(window)) % world
        cols = (1 - window // 2 + np.arange(window)) % world
        expected = np.minimum(canvas[np.ix_(rows, cols)], 255).astype(np.uint8)
        frame = env.render()[0]
        assert (frame[..., 0] == frame[..., 2]).all()
        np.testing.assert_array_equal(frame[..., 0].numpy(), expected)
        assert (canvas > 255).any()  # the clip is exercised


def test_render_of_a_state_matches_the_live_frames_and_leaves_the_lanes_alone():
    with _make(num_envs=3) as env:
        env.reset(seed=SEED)
        env.step(_actions(4, 1, lanes=3)[0], frames=None)
        state = env.get_state()
        live = env.render()
        torch.testing.assert_close(env.render(state), live, rtol=0, atol=0)
        stacked = torch.cat([state[1:2], state[0:1], state[1:2]])
        torch.testing.assert_close(env.render(stacked), live[[1, 0, 1]], rtol=0, atol=0)
        assert_same_state(env.get_state(), state)
        assert live.shape == (3, 32, 32, 3) and live.dtype == torch.uint8
        assert live.float().std() > 10


def test_finished_lanes_hold_and_episodes_truncate_at_the_limit():
    with _make(max_episode_steps=3) as env:
        env.reset(seed=SEED)
        out = env.step(_actions(5, 3).transpose(1, 0, 2), states=slice(None))
        assert out["truncated"][:, -1].all() and not out["terminated"].any()
        assert out["state"]["step"].tolist() == [[3], [3]]
        out = env.step(_actions(6, 2).transpose(1, 0, 2))
        assert out["valid"][:, 0].all() and not out["valid"][:, 1].any()
        assert out["state"]["step"].tolist() == [[4], [4]]
    with _make(max_episode_steps=None) as env:
        env.reset(seed=SEED)
        assert not env.step(_actions(7, 1)[0])["truncated"].any()


def test_a_masked_step_leaves_inactive_lanes_alone():
    with _make() as env:
        before = env.reset(seed=SEED)
        after = env._step(_actions(8, 1)[0], active=torch.tensor([False, True]))["state"]
        assert_same_state(after[0], before[0])
        assert after["step"].tolist() == [[0], [1]]


def _bad(state, key, value):
    bad = state.clone()
    bad[key] = value
    return bad


def test_invalid_states_are_rejected_before_any_lane_changes():
    with _make() as env:
        env.reset(seed=SEED)
        good = env.get_state()
        before = env.get_state()
        label = good["digits", "label"].clone()
        label[1, 0] = (label[1, 0] + 1) % 10
        position = good["digits", "position"].clone()
        position[1, 3] = 50
        cases = [
            _bad(good, ("digits", "label"), label),
            _bad(good, ("digits", "position"), position),
            _bad(good, ("digits", "image"), torch.full_like(good["digits", "image"], 60000)),
            _bad(good, ("digits", "velocity"), torch.full_like(good["digits", "velocity"], 2)),
            _bad(good, ("camera",), good["camera"].float()),
            _bad(good, ("step",), torch.full_like(good["step"], -1)),
        ]
        missing = good.clone()
        del missing["rng", "episode"]
        cases.append(missing)
        for bad in cases:
            with pytest.raises((TypeError, ValueError)):
                env.set_state(bad)
            assert_same_state(env.get_state(), before)
        with pytest.raises(ValueError):
            env.set_state(good[:1])
        with pytest.raises(ValueError):
            env.set_state(good["digits"])


def test_actions_must_be_integer_offsets_in_range():
    with _make() as env:
        env.reset(seed=SEED)
        before = env.get_state()
        for bad in (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0, 11], [0, 0]]), np.array([[0, 0], [-11, 0]])):
            with pytest.raises(ValueError):
                env.step(bad)
            assert_same_state(env.get_state(), before)
        assert env.action_spec.low == -10 and env.action_spec.high == 10
    with _make(preset="dynamic_fo_no_sm") as env:
        env.step(np.zeros((2, 2), np.int64))
        with pytest.raises(ValueError):
            env.step(np.array([[0, 1], [0, 0]]))


@pytest.mark.parametrize(
    ("preset", "world", "digits", "motion", "moving"),
    [
        ("dynamic_po", 50, 5, 10, True),
        ("static_po", 50, 5, 10, False),
        ("dynamic_fo", 32, 3, 10, True),
        ("dynamic_fo_no_sm", 32, 3, 0, True),
    ],
)
def test_presets_follow_the_paper(preset, world, digits, motion, moving):
    with _make(num_envs=8, preset=preset) as env:
        state = env.reset(seed=SEED)
        assert env.config["world_size"] == world and env.config["window_size"] == 32
        assert state["digits", "image"].shape == (8, digits)
        assert env.action_spec.high == motion
        velocity = state["digits", "velocity"]
        assert velocity.abs().max() <= 1 and bool(velocity.any()) == moving
        assert int(state["digits", "position"].max()) < world


def test_the_default_is_the_main_result_of_the_paper():
    with _make() as env:
        assert env.preset == "dynamic_po"


def test_options_are_checked():
    with pytest.raises(TypeError, match="unknown MNIST World option"):
        _make(digits=3)
    with pytest.raises(ValueError, match="preset"):
        _make(preset="dynamic")
    with pytest.raises(ValueError, match="world_size"):
        _make(world_size=20)
    with pytest.raises(ValueError):
        _make(size=64)
    with _make(num_envs=1, world_size=80, num_digits=8, max_speed=2) as env:
        state = env.reset(seed=SEED)
        assert state["digits", "position"].shape == (1, 16)
        assert int(state["digits", "velocity"].abs().max()) <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_lanes_match_cpu_lanes():
    with _make() as cpu, snv.MNISTWorld(num_envs=2, device="cuda") as gpu:
        state = cpu.reset(seed=SEED)
        gpu.set_state(state.to("cuda"))
        for action in _actions(9, 4):
            want = cpu.step(action)
            got = gpu.step(torch.as_tensor(action, device="cuda"))
            assert_same_state(got.to("cpu"), want)


def test_close_is_idempotent_and_refuses_later_calls():
    env = _make(num_envs=1)
    env.close()
    env.close()
    with pytest.raises(RuntimeError):
        env.reset(seed=0)
    with pytest.raises(RuntimeError):
        env.render()
