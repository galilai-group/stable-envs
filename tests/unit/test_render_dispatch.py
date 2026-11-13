"""``Env.render`` routes None, a full State, a RenderState and compact rows, and checks each input once."""

import numpy as np
import pytest
import torch

import stable_envs as se
import stable_envs.env as env_module


def _count_scans(monkeypatch):
    calls = []
    scan = env_module._nonfinite_report

    def counted(*args, **kwargs):
        calls.append(1)
        return scan(*args, **kwargs)

    monkeypatch.setattr(env_module, "_nonfinite_report", counted)
    return calls


def test_each_input_form_draws_the_same_frames():
    with se.make("CartPole", num_envs=3, device="cpu") as env:
        state = env.reset(seed=5)
        rows = env._pack_render_state(state)
        frames = env.render(state)
        assert frames.dtype == torch.uint8 and frames.shape == (3, *env.frame_shape, 3)
        assert torch.equal(env.render(), frames)
        assert torch.equal(env.render(rows), frames)
        assert torch.equal(env.render(rows.numpy()), frames)
        assert torch.equal(env.render(rows.double()), frames)
        assert torch.equal(env.render(rows[None]), frames[None])
        sampled = env.sample_states(4, np.random.default_rng(0))
        assert env.render(sampled).shape == (4, *env.frame_shape, 3)


def test_a_state_and_its_packed_rows_draw_identical_frames_under_one_variation():
    with se.make("CartPole", num_envs=4, device="cpu") as env:
        env.set_variation(env.sample_variation(4, rng=np.random.default_rng(1)))
        state = env.reset(seed=2)
        for _ in range(3):
            state = env.step(env.action_spec.sample(4, np.random.default_rng(3)), frames=None)["state"]
        assert torch.equal(env.render(state), env.render(env._pack_render_state(state)))


def test_rows_are_checked_before_drawing():
    with se.make("CartPole", num_envs=2, device="cpu") as env:
        rows = env._pack_render_state(env.reset(seed=0)).clone()
        with pytest.raises(ValueError, match="state_dim"):
            env.render(rows[:, :-1])
        with pytest.raises(ValueError, match="FRAMES"):
            env.render(torch.zeros((2, 8, 8, 3), dtype=torch.uint8))
        rows[1, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            env.render(rows)


def test_render_scans_rows_for_nonfinite_values_once(monkeypatch):
    with se.make("CartPole", num_envs=2, device="cpu") as env:
        state = env.reset(seed=0)
        rows = env._pack_render_state(state).numpy()
        calls = _count_scans(monkeypatch)
        env.render(rows)
        assert len(calls) == 1
        calls.clear()
        env.render(state)
        assert calls == []
        assert env._checked_rows is None


@pytest.mark.family("pusht")
@pytest.mark.parametrize("name", ["PushT", "PushTDiscrete"])
def test_pusht_draws_a_render_state_and_refuses_rows(name):
    with se.make(name, num_envs=2, device="cpu") as env:
        state = env.reset(seed=4)
        assert torch.equal(env.render(env.to_render_state(state)), env.render(state))
        with pytest.raises(TypeError) as e:
            env.render(np.zeros((2, 5), np.float32))
        msg = str(e.value)
        assert "no compact rows" in msg and "full State" in msg and "to_render_state()" in msg


@pytest.mark.family("miniworld")
def test_miniworld_refuses_rows():
    with se.make("MiniWorldMovingBlockWorld", num_envs=1, device="cpu") as env:
        with pytest.raises(TypeError, match="no compact rows.*full State"):
            env.render(np.zeros((1, 4), np.float32))


@pytest.mark.family("craftax")
def test_craftax_refuses_rows():
    with se.make("CraftaxClassicPixels", num_envs=1, device="cpu") as env:
        with pytest.raises(TypeError, match="no compact rows.*full State"):
            env.render(np.zeros((1, 4), np.float32))


@pytest.mark.family("rocket")
def test_rocket_on_cpu_refuses_rows_and_names_the_accepted_forms():
    with se.make("PyFlytRocketLanding", num_envs=1, device="cpu") as env:
        rows = env.sample_states(1, np.random.default_rng(0))
        with pytest.raises(TypeError) as e:
            env.render(rows)
        msg = str(e.value)
        assert "only on CUDA" in msg and "full State" in msg and "to_render_state()" in msg
        assert "rows as sample_states()" not in msg


@pytest.mark.family("dmc")
def test_dmc_state_draws_its_own_appearance_and_rows_the_current_variation():
    pytest.importorskip("stable_envs.envs.dmcontrol")
    from stable_envs._core.physics import state_key

    with se.make("DMCBallInCupCatch", num_envs=2, device="cpu") as env:
        env.set_variation({"ball.size": 0.04})
        state = env.reset(seed=0)
        size = state[state_key("ball.size")].to(torch.float32)
        assert torch.allclose(size, torch.full_like(size, 0.04))
        env.set_variation({"ball.size": 0.015})
        env.reset(seed=0)
        rows = env._pack_render_state(state)
        own = env.render(state)
        current = env.render(rows)
        assert not torch.equal(own, current)
        drawn = {p: state[state_key(p)].to(torch.float32) for p in env.FRAME_PHYSICS}
        assert torch.equal(own, env.render(rows, variation=drawn))
