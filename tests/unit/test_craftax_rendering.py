"""Delegated Craftax pixel tasks render the same frames as the stable-worldmodel oracle."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

import stable_envs as se
from tests.helpers.reference.craftax_source import ACTIONS, assert_state_equal, fixture, oracle, restore_oracle

DEVICE = os.environ.get("CRAFTAX_DEVICE", "cpu")
PIXELS = ("CraftaxClassicPixels", "CraftaxPixels")


@pytest.mark.nightly
@pytest.mark.parametrize("name", PIXELS)
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_initial_and_trajectory_source_frames(name, seed):
    source = oracle(name)
    with se.make(name, device=DEVICE) as env:
        source.reset(seed=seed)
        env.reset(seed=seed)
        for action in [None, 1, 4, 5, 0, 2, 0, 6]:
            if action is not None:
                source.step(action)
                env.step([action], frames=None)
            saved = env.get_state()
            expected = source.render()
            actual = env.render(saved)[0].cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
            draw = env.to_render_state(saved)
            assert torch.equal(env.render(saved), env.render(draw))
            assert_state_equal(saved, env.get_state())


@pytest.mark.nightly
@pytest.mark.parametrize("name", PIXELS)
@pytest.mark.parametrize("case", list(ACTIONS))
def test_gameplay_source_frames_and_nonmutation(name, case, monkeypatch):
    source = oracle(name)
    with se.make(name, device=DEVICE) as env:
        env.reset(seed=17)
        source.reset(seed=17)
        snapshot = fixture(env, name, case)
        restore_oracle(source, env, snapshot)
        env.set_state(snapshot)
        if not bool(snapshot["lifecycle", "terminated"][0]):
            source.step(ACTIONS[case])
            env.step([ACTIONS[case]], frames=None)
            expected = source.render()
        else:
            source._last_obs = np.asarray(jax.jit(source._env.get_obs)(source._state))
            expected = source.render()
        full = env.get_state()
        draw = env.to_render_state(full)
        frozen_draw = draw.clone()

        def forbidden(*args, **kwargs):
            raise AssertionError("renderer must not reset or step")

        monkeypatch.setattr(env._simulator, "step_fn", forbidden)
        monkeypatch.setattr(env._simulator, "reset_fn", forbidden)
        rng = np.random.get_state()
        frame = env.render(full)
        np.testing.assert_array_equal(frame[0].cpu().numpy(), expected)
        assert torch.equal(frame, env.render(draw))
        assert torch.equal(frame, env.render(draw))
        assert_state_equal(full, env.get_state())
        assert_state_equal(draw, frozen_draw)
        after = np.random.get_state()
        assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
        with pytest.raises(ValueError):
            env.set_state(draw)


@pytest.mark.parametrize("name", PIXELS)
def test_batch_chunk_order_independent_instance_and_validation(name):
    with se.make(name, num_envs=3, device=DEVICE) as env, se.make(name, num_envs=3, device=DEVICE) as other:
        full = env.reset(seed=17)
        other.reset(seed=1)
        before = other.get_state()
        draw = env.to_render_state(full)
        image = env.render(full)
        assert torch.equal(image, env.render(draw, chunk_size=1))
        assert torch.equal(image, env.render(draw, chunk_size=2))
        order = torch.tensor([2, 0, 1], device=DEVICE)
        assert torch.equal(image[order], env.render(draw[order]))
        assert torch.equal(image, other.render(draw))
        assert_state_equal(before, other.get_state())
        target = torch.empty_like(image)
        assert env.render(draw, out=target) is target
        assert torch.equal(target, image)
        bad = draw.clone()
        bad["renderer", "signature"][2, 0] += 1
        with pytest.raises(ValueError):
            env.render(bad)
        for kwargs in [{"size": 16}, {"supersample": 2}, {"backend": "mjwarp"}, {"chunk_size": 0}]:
            with pytest.raises(ValueError):
                env.render(draw, **kwargs)
        with pytest.raises(ValueError):
            env.render(draw, out=torch.zeros(1))
        assert_state_equal(full, env.get_state())


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxSymbolic"])
def test_symbolic_source_frame_refusal(name):
    with se.make(name, device=DEVICE) as env:
        env.reset(seed=0)
        with pytest.raises(NotImplementedError, match="symbolic"):
            env.render()
        with pytest.raises(NotImplementedError, match="symbolic"):
            env.to_render_state()


@pytest.mark.parametrize("name", PIXELS)
def test_source_day_cycle_and_sleep_exact(name, tmp_path):
    import importlib

    source = oracle(name)
    module = importlib.import_module(
        "craftax." + ("craftax_classic" if "Classic" in name else "craftax") + ".game_logic"
    )
    with se.make(name, device=DEVICE) as env:
        source.reset(seed=17)
        original = env.reset(seed=17)
        for timestep in [0, 37, 74, 75, 112, 149, 150, 187, 224, 225, 262, 299]:
            for sleeping in [False, True]:
                state = original.clone()
                state["engine", "timestep"].fill_(timestep)
                state["engine", "is_sleeping"].fill_(sleeping)
                light = float(module.calculate_light_level(timestep, source._params))
                state["engine", "light_level"].fill_(light)
                env.set_state(state)
                restore_oracle(source, env, state)
                source.step(0)
                after = env.step([0], frames=None)["state"]
                expected, actual = source.render(), env.render(after)[0].cpu().numpy()
                if not np.array_equal(expected, actual):
                    path = tmp_path / (name + "-day-cycle-mismatch.npz")
                    np.savez_compressed(
                        path,
                        expected=expected,
                        actual=actual,
                        **{"state/" + "/".join(k): v.cpu().numpy() for k, v in after.items(True, True)},
                    )
                    pytest.fail(f"Exact source frame mismatch at timestep={timestep}, sleeping={sleeping}: {path}")
