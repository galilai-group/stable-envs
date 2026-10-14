"""Policies and ``tests.helpers.rollout.sample`` on CPU, against a stand-in env with a known answer.

The stand-in integrates ``state' = state + action``, so the trajectory is known in closed form and
"these are the actions that produced these states" is an exact assertion. The checks cover seeding,
recorded actions, the ``t=0`` state, and states that must not alias the env's buffer.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.helpers.rollout import BasePolicy, RandomPolicy, ScriptedPolicy, sample


class RenderOnlyEnv:
    """An env with a renderer and no batched simulator."""

    state_dim = 3
    style_dim = 0

    def __init__(self, batch: int = 4):
        self.B = int(batch)
        self.size = 4
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(0)
        self._render_state = torch.zeros(self.B, self.state_dim)

    def sample_states(self, batch, rng=None):
        r = rng or self.rng
        return r.uniform(-1, 1, (int(batch), self.state_dim)).astype(np.float32)

    def _get_render_state(self):
        return self._render_state

    def get_style(self):
        return torch.zeros(self.B, 0)

    def _set_render_state(self, states, style=None):
        self._render_state.copy_(torch.as_tensor(states, dtype=torch.float32).reshape(self.B, -1))
        return self

    def _render(self, states=None, **kw):
        s = torch.as_tensor(self._render_state if states is None else states, dtype=torch.float32).reshape(
            -1, self.state_dim
        )
        img = torch.zeros((s.shape[0], 2, 3, 3), dtype=torch.uint8)
        img[:, 0, 0, :] = (s[:, 0].to(torch.int64) % 251).to(torch.uint8).unsqueeze(-1)
        return img


class IntegratorEnv(RenderOnlyEnv):
    """The same, plus a batched GPU-shaped simulator whose trajectory is known in closed form."""

    action_dim = 3
    action_low = -1.0
    action_high = 1.0

    def step(self, actions, *, states=slice(None), frames=slice(-1, None)):
        self._render_state += torch.as_tensor(actions, dtype=torch.float32).reshape(self.B, -1)
        return self._render_state


class DiscreteEnv(IntegratorEnv):
    """Half-open integer bounds ``[0, 7)``."""

    action_dim = 1
    action_low = 0.0
    action_high = 7.0


# ------------------------------------------------------------------ the refusals


def test_an_env_with_no_simulator_says_so_and_points_at_the_alternative():
    """``sample`` on a renderer-only env raises ``NotImplementedError`` naming the env and the alternative."""
    with pytest.raises(NotImplementedError) as e:
        sample(RenderOnlyEnv(), steps=2)
    msg = str(e.value)
    assert "RenderOnlyEnv" in msg
    assert "states you already have" in msg, f"the refusal must name what to do instead: {msg}"


def test_a_batch_that_is_not_the_env_s_batch_is_refused_at_the_top():
    """Every buffer in the loop is sized ``env.B``; a mismatch would fail deep or, worse, broadcast."""
    with pytest.raises(ValueError, match="batch 8 != env batch 4"):
        sample(IntegratorEnv(batch=4), steps=2, batch=8)


# ------------------------------------------------------------------ what sample returns


def test_the_trajectory_has_one_more_state_than_it_has_actions():
    """``steps+1`` states and ``steps`` actions: the initial state is recorded before the first step."""
    env = IntegratorEnv(batch=4)
    tr = sample(env, steps=5, seed=0)
    assert tr["states"].shape == (4, 6, 3)
    assert tr["actions"].shape == (4, 5, 3)
    assert tr["style"].shape == (4, 0)


def test_the_returned_actions_ARE_the_actions_the_simulator_was_driven_with():
    """``state[t+1] - state[t]`` equals ``action[t]`` for the integrator."""
    env = IntegratorEnv(batch=4)
    tr = sample(env, steps=6, seed=1)
    delta = tr["states"][:, 1:] - tr["states"][:, :-1]
    assert torch.allclose(delta, tr["actions"], atol=1e-6), "the stored actions do not account for the stored states"


def test_each_recorded_state_is_a_COPY_and_not_the_env_s_live_buffer():
    """Without a copy, every stored step would alias one tensor holding the last state."""
    env = IntegratorEnv(batch=4)
    tr = sample(env, steps=4, seed=2)
    assert not torch.allclose(tr["states"][:, 0], tr["states"][:, -1]), (
        "the first and last states are identical, which for a random-driven integrator means every "
        "recorded step is the same aliased buffer"
    )


def test_init_states_are_used_verbatim_as_step_zero():
    """The entry point for continuing a trajectory, or for rolling out from a stored state."""
    env = IntegratorEnv(batch=4)
    init = np.arange(12, dtype=np.float32).reshape(4, 3)
    tr = sample(env, steps=3, seed=0, init_states=init)
    assert torch.allclose(tr["states"][:, 0], torch.as_tensor(init))


def test_frames_are_OFF_by_default_and_correct_when_asked_for():
    """No frames by default; when requested, frame ``t`` is the render of state ``t``."""
    env = IntegratorEnv(batch=4)
    assert "frames" not in sample(env, steps=3, seed=0)

    tr = sample(env, steps=3, seed=0, render=True)
    assert tr["frames"].shape == (4, 4, 2, 3, 3)
    for t in range(4):
        want = env._render(tr["states"][:, t], style=tr["style"])
        assert torch.equal(tr["frames"][:, t], want), f"frame {t} is not the render of state {t}"


def test_a_seed_makes_the_whole_rollout_reproducible():
    """Same seed, same trajectory; different seed, different trajectory (so the seed is not ignored)."""
    env = IntegratorEnv(batch=4)
    a = sample(env, steps=5, seed=42)["states"]
    b = sample(env, steps=5, seed=42)["states"]
    c = sample(env, steps=5, seed=43)["states"]
    assert torch.equal(a, b), "the same seed gave two different rollouts"
    assert not torch.equal(a, c), "two different seeds gave the same rollout, so the seed is ignored"


def test_the_seed_reaches_the_env_s_own_generator_too():
    """``sample_states`` draws the initial condition, so a seeded rollout must seed that as well."""
    env = IntegratorEnv(batch=4)
    first = sample(env, steps=1, seed=7)["states"][:, 0].clone()
    env.rng = np.random.default_rng(999)  # advance it somewhere else entirely
    again = sample(env, steps=1, seed=7)["states"][:, 0]
    assert torch.equal(first, again), "the initial states were not drawn from the rollout's seed"


# ------------------------------------------------------------------ the policies


def test_the_base_policy_is_an_interface_and_says_so():
    """``get_action`` must raise rather than return None, which would fail three frames later."""
    p = BasePolicy()
    assert p.env is None
    env = IntegratorEnv()
    p.set_env(env)
    assert p.env is env
    assert p.reset(4) is None, "reset is a hook for stateful policies, not a value"
    with pytest.raises(NotImplementedError):
        p.get_action(env._get_render_state())


def test_the_random_policy_adopts_the_env_s_declared_bounds():
    """With no arguments, the policy reads the env's declared bounds."""
    env = IntegratorEnv()
    p = RandomPolicy(seed=0)
    p.set_env(env)
    assert (p.low, p.high) == (env.action_low, env.action_high)


def test_the_random_policy_stays_inside_the_bounds_including_half_open_discrete_ones():
    """Draws stay in ``[low, high)``; the high bound is exclusive for discrete envs."""
    env = DiscreteEnv()
    p = RandomPolicy(seed=3)
    p.set_env(env)
    p.reset(env.B)
    for _ in range(50):
        a = p.get_action(env._get_render_state())
        assert a.shape == (env.B, env.action_dim)
        assert float(a.min()) >= env.action_low
        assert float(a.max()) < env.action_high, "a half-open bound was reached"


def test_the_random_policy_re_seeds_on_reset_so_two_rollouts_agree():
    """``reset`` re-seeds, so one policy object gives the same actions after each reset."""
    env = IntegratorEnv()
    p = RandomPolicy(seed=11)
    p.set_env(env)
    p.reset(env.B)
    first = [p.get_action(env._get_render_state()).clone() for _ in range(4)]
    p.reset(env.B)
    second = [p.get_action(env._get_render_state()).clone() for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second)), (
        "reset() did not put the policy's generator back, so the second rollout of a seeded pair "
        "differs from the first"
    )


def test_the_random_policy_honours_bounds_given_explicitly_over_the_env_s():
    env = IntegratorEnv()
    p = RandomPolicy(low=0.25, high=0.5, seed=0)
    p.set_env(env)
    p.reset(env.B)
    a = p.get_action(env._get_render_state())
    assert float(a.min()) >= 0.25 and float(a.max()) <= 0.5


def test_the_random_policy_draws_integer_actions_for_an_integer_action_spec():
    """CartPole's spec is int64 in [0, 1]; float draws are rejected by the upstream environment."""
    import stable_envs as snv

    env = snv.CartPole(num_envs=8, device="cpu")
    try:
        tr = sample(env, steps=6, seed=0)
        a = tr["actions"]
        assert a.dtype == torch.int64
        assert set(torch.unique(a).tolist()) <= {0, 1}
        p = RandomPolicy(seed=0)
        p.set_env(env)
        p.set_env(env)  # a second call must keep the spec's range, not the continuous defaults
        assert (p.low, p.high) == (0, 1)
    finally:
        env.close()


def test_the_discrete_flag_overrides_the_action_spec():
    env = IntegratorEnv()
    p = RandomPolicy(low=0, high=3, seed=0, discrete=True)
    p.set_env(env)
    a = p.get_action(env._get_render_state())
    assert not a.is_floating_point() and int(a.min()) >= 0 and int(a.max()) <= 3

    import stable_envs as snv

    cartpole = snv.CartPole(num_envs=4, device="cpu")
    try:
        p = RandomPolicy(seed=0, discrete=False)
        p.set_env(cartpole)
        assert p.get_action(cartpole._get_render_state()).is_floating_point()
    finally:
        cartpole.close()


def test_a_scripted_policy_is_given_the_state_and_the_timestep():
    """``ScriptedPolicy`` calls ``fn(state, t)`` with an advancing timestep."""
    seen = []

    def fn(state, t):
        seen.append((state.clone(), t))
        return torch.full((state.shape[0], 3), float(t))

    env = IntegratorEnv(batch=4)
    tr = sample(env, steps=3, policy=ScriptedPolicy(fn), seed=0)
    assert [t for _, t in seen] == [0, 1, 2], "the timestep is not being passed, or not advancing"
    assert torch.allclose(tr["actions"][:, 2], torch.full((4, 3), 2.0))
    delta = tr["states"][:, 1:] - tr["states"][:, :-1]
    assert torch.allclose(delta, tr["actions"], atol=1e-6)


def test_a_policy_gets_set_env_and_reset_before_the_first_action():
    """``set_env`` precedes ``reset``, which precedes the first action."""
    calls = []

    class _Recording(BasePolicy):
        def set_env(self, env):
            calls.append("set_env")
            self.env = env

        def reset(self, batch):
            calls.append(f"reset:{batch}")

        def get_action(self, obs, **kw):
            calls.append("action")
            return torch.zeros(obs.shape[0], 3)

    sample(IntegratorEnv(batch=4), steps=2, policy=_Recording(), seed=0)
    assert calls[:2] == ["set_env", "reset:4"], calls
    assert calls.count("action") == 2


def test_env_does_not_offer_a_policy_sample_shorthand():
    """Policy rollouts are not an environment API; ``tests.helpers.rollout.sample`` provides them."""
    from stable_envs.env import Env

    assert "sample" not in Env.__dict__
