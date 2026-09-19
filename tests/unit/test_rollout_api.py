"""Policies and ``sample``: the contract, on CPU, with a simulator whose answer is known by hand.

``stable_envs/rollout.py`` is the half of the repo that answers "I have no states yet". It had no direct
test — ``tests/gpu/test_reproducibility.py`` asserts that ``env.sample(seed=)`` reproduces in a fresh
interpreter, which is the strongest thing to say about it and says nothing about what it RETURNS. The
failures below all produce a dict of tensors with the right keys and the right shapes:

  * a policy whose ``reset`` does not re-seed makes the second rollout of a seeded pair differ from the
    first, and only when something else consumed from the generator in between;
  * ``actions`` that are not the actions the simulator was driven with give a corpus whose transitions
    do not hold, which no shape or dtype check can see;
  * a step applied before the initial state is recorded loses ``t=0``, so every stored trajectory is
    off by one against the actions beside it;
  * ``states`` that alias the env's own buffer come back as ``steps+1`` copies of the final state.

The stand-in env integrates: ``state' = state + action``. That makes the whole trajectory predictable in
closed form, so "are these the actions that produced these states" is an assertion rather than a
plausibility check, and it needs no GPU, no renderer and no reference simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.rollout import BasePolicy, RandomPolicy, ScriptedPolicy, sample


class RenderOnlyEnv:
    """An env that ships a renderer and NO batched simulator. Most of the registry is this."""

    state_dim = 3
    style_dim = 0

    def __init__(self, batch: int = 4):
        self.B = int(batch)
        self.size = 4
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(0)
        self._state = torch.zeros(self.B, self.state_dim)

    def sample_states(self, batch, rng=None):
        r = rng or self.rng
        return r.uniform(-1, 1, (int(batch), self.state_dim)).astype(np.float32)

    def get_state(self):
        return self._state

    def get_style(self):
        return torch.zeros(self.B, 0)

    def set_state(self, states, style=None):
        self._state.copy_(torch.as_tensor(states, dtype=torch.float32).reshape(self.B, -1))
        return self

    def render(self, states=None, **kw):
        s = torch.as_tensor(self._state if states is None else states, dtype=torch.float32).reshape(-1, self.state_dim)
        img = torch.zeros((s.shape[0], 2, 3, 3), dtype=torch.uint8)
        img[:, 0, 0, :] = (s[:, 0].to(torch.int64) % 251).to(torch.uint8).unsqueeze(-1)
        return img


class IntegratorEnv(RenderOnlyEnv):
    """The same, plus a batched GPU-shaped simulator whose trajectory is known in closed form."""

    action_dim = 3
    action_low = -1.0
    action_high = 1.0

    def step(self, actions):
        self._state += torch.as_tensor(actions, dtype=torch.float32).reshape(self.B, -1)
        return self._state


class DiscreteEnv(IntegratorEnv):
    """Half-open integer bounds, as MiniGrid's seven actions and CartPole's two declare them."""

    action_dim = 1
    action_low = 0.0
    action_high = 7.0


# ------------------------------------------------------------------ the refusals


def test_an_env_with_no_simulator_says_so_and_points_at_the_alternative():
    """The split is deliberate, so the message has to read as a design statement and not as a gap.

    Rendering and simulation are separable and an env earns a ``step()`` only where its dynamics can be
    ported AND checked, so most of the registry has none. Without this the failure is an
    ``AttributeError: 'X' object has no attribute 'step'`` from inside a loop, which reads as a bug in
    the library.
    """
    with pytest.raises(NotImplementedError) as e:
        sample(RenderOnlyEnv(), steps=2)
    msg = str(e.value)
    assert "RenderOnlyEnv" in msg
    assert "StateDataset" in msg and "render_states" in msg, (
        f"the refusal must name what to do instead — supply states you already have: {msg}"
    )


def test_a_batch_that_is_not_the_env_s_batch_is_refused_at_the_top():
    """Every buffer in the loop is sized ``env.B``; a mismatch would fail deep or, worse, broadcast."""
    with pytest.raises(ValueError, match="batch 8 != env batch 4"):
        sample(IntegratorEnv(num_envs=4), steps=2, batch=8)


# ------------------------------------------------------------------ what sample returns


def test_the_trajectory_has_one_more_state_than_it_has_actions():
    """``steps+1`` states and ``steps`` actions: the initial state is recorded BEFORE the first step.

    Off by one in the other direction is the tempting implementation (step, then record) and it loses
    ``t=0`` entirely, so every stored trajectory is shifted against the actions stored beside it and
    the corpus teaches a transition nobody took.
    """
    env = IntegratorEnv(num_envs=4)
    tr = sample(env, steps=5, seed=0)
    assert tr["states"].shape == (4, 6, 3)
    assert tr["actions"].shape == (4, 5, 3)
    assert tr["style"].shape == (4, 0)


def test_the_returned_actions_ARE_the_actions_the_simulator_was_driven_with():
    """The integrator makes this exact: ``state[t+1] - state[t]`` must equal ``action[t]``.

    A rollout that returns a policy's PROPOSED actions while stepping with something else — a clipped
    copy, a reshaped one, the next timestep's — produces a corpus whose transitions simply do not hold,
    and every shape, dtype and range check passes on it.
    """
    env = IntegratorEnv(num_envs=4)
    tr = sample(env, steps=6, seed=1)
    delta = tr["states"][:, 1:] - tr["states"][:, :-1]
    assert torch.allclose(delta, tr["actions"], atol=1e-6), "the stored actions do not account for the stored states"


def test_each_recorded_state_is_a_COPY_and_not_the_env_s_live_buffer():
    """Without the clone, every stored step aliases one tensor and the corpus is the last state, N times.

    Silent: the shape is right, the dtype is right, and a loss computed on it converges to something.
    """
    env = IntegratorEnv(num_envs=4)
    tr = sample(env, steps=4, seed=2)
    assert not torch.allclose(tr["states"][:, 0], tr["states"][:, -1]), (
        "the first and last states are identical, which for a random-driven integrator means every "
        "recorded step is the same aliased buffer"
    )


def test_init_states_are_used_verbatim_as_step_zero():
    """The entry point for continuing a trajectory, or for rolling out from a stored state."""
    env = IntegratorEnv(num_envs=4)
    init = np.arange(12, dtype=np.float32).reshape(4, 3)
    tr = sample(env, steps=3, seed=0, init_states=init)
    assert torch.allclose(tr["states"][:, 0], torch.as_tensor(init))


def test_frames_are_OFF_by_default_and_correct_when_asked_for():
    """707 MB of pixels against 512 kB of states, and the states re-render at any resolution later.

    Materialising frames during collection re-creates exactly the problem this repo exists to remove,
    so the default is the assertion. When they ARE asked for, frame ``t`` has to be the render of state
    ``t`` — the loop writes them one timestep at a time into a preallocated buffer, which is where an
    off-by-one or a stale row lives.
    """
    env = IntegratorEnv(num_envs=4)
    assert "frames" not in sample(env, steps=3, seed=0)

    tr = sample(env, steps=3, seed=0, render=True)
    assert tr["frames"].shape == (4, 4, 2, 3, 3)
    for t in range(4):
        want = env.render(tr["states"][:, t], style=tr["style"])
        assert torch.equal(tr["frames"][:, t], want), f"frame {t} is not the render of state {t}"


def test_a_seed_makes_the_whole_rollout_reproducible():
    """Same seed, same trajectory; different seed, different trajectory.

    The second half is not padding: a rollout that ignores the seed entirely satisfies the first half
    perfectly, and that is the failure mode — a policy that seeds on ``set_env`` but never re-seeds
    looks reproducible until something else advances the generator.
    """
    env = IntegratorEnv(num_envs=4)
    a = sample(env, steps=5, seed=42)["states"]
    b = sample(env, steps=5, seed=42)["states"]
    c = sample(env, steps=5, seed=43)["states"]
    assert torch.equal(a, b), "the same seed gave two different rollouts"
    assert not torch.equal(a, c), "two different seeds gave the same rollout, so the seed is ignored"


def test_the_seed_reaches_the_env_s_own_generator_too():
    """``sample_states`` draws the initial condition, so a seeded rollout must seed that as well."""
    env = IntegratorEnv(num_envs=4)
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
        p.get_action(env.get_state())


def test_the_random_policy_adopts_the_env_s_declared_bounds():
    """No arguments needed, because the bounds are the env's to declare — and it must READ them."""
    env = IntegratorEnv()
    p = RandomPolicy(seed=0)
    p.set_env(env)
    assert (p.low, p.high) == (env.action_low, env.action_high)


def test_the_random_policy_stays_inside_the_bounds_including_half_open_discrete_ones():
    """``[0, n)`` and floor, so a uniform float is a uniform action and an integer tensor still works.

    The high bound is EXCLUSIVE for the discrete envs (MiniGrid's seven actions, CartPole's two), and a
    policy that could return exactly ``n`` would index one past the end of a transition table.
    """
    env = DiscreteEnv()
    p = RandomPolicy(seed=3)
    p.set_env(env)
    p.reset(env.B)
    for _ in range(50):
        a = p.get_action(env.get_state())
        assert a.shape == (env.B, env.action_dim)
        assert float(a.min()) >= env.action_low
        assert float(a.max()) < env.action_high, "a half-open bound was reached"


def test_the_random_policy_re_seeds_on_reset_so_two_rollouts_agree():
    """THE property that makes a seeded rollout reproducible more than once in a process.

    Without it the generator carries on from wherever the previous rollout left it, so the first
    rollout after construction reproduces and the second does not — which is the shape of bug that
    survives every test written as "run it twice with the same seed" if that test constructs a fresh
    policy each time. This one deliberately reuses ONE policy object.
    """
    env = IntegratorEnv()
    p = RandomPolicy(seed=11)
    p.set_env(env)
    p.reset(env.B)
    first = [p.get_action(env.get_state()).clone() for _ in range(4)]
    p.reset(env.B)
    second = [p.get_action(env.get_state()).clone() for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second)), (
        "reset() did not put the policy's generator back, so the second rollout of a seeded pair "
        "differs from the first"
    )


def test_the_random_policy_honours_bounds_given_explicitly_over_the_env_s():
    env = IntegratorEnv()
    p = RandomPolicy(low=0.25, high=0.5, seed=0)
    p.set_env(env)
    p.reset(env.B)
    a = p.get_action(env.get_state())
    assert float(a.min()) >= 0.25 and float(a.max()) <= 0.5


def test_a_scripted_policy_is_given_the_state_and_the_timestep():
    """``fn(state, t)``. The timestep is what makes an open-loop schedule expressible at all."""
    seen = []

    def fn(state, t):
        seen.append((state.clone(), t))
        return torch.full((state.shape[0], 3), float(t))

    env = IntegratorEnv(num_envs=4)
    tr = sample(env, steps=3, policy=ScriptedPolicy(fn), seed=0)
    assert [t for _, t in seen] == [0, 1, 2], "the timestep is not being passed, or not advancing"
    assert torch.allclose(tr["actions"][:, 2], torch.full((4, 3), 2.0))
    delta = tr["states"][:, 1:] - tr["states"][:, :-1]
    assert torch.allclose(delta, tr["actions"], atol=1e-6)


def test_a_policy_gets_set_env_and_reset_before_the_first_action():
    """Order matters: a stateful policy sizes its buffers in ``reset`` and needs the env before that."""
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

    sample(IntegratorEnv(num_envs=4), steps=2, policy=_Recording(), seed=0)
    assert calls[:2] == ["set_env", "reset:4"], calls
    assert calls.count("action") == 2


def test_an_env_level_sample_is_the_module_level_one():
    """``env.sample(...)`` is the ergonomic entry point and must not be a second implementation."""
    from stable_envs.env import Env as StableEnv

    assert "sample" in StableEnv.__dict__, "StableEnv no longer offers the shorthand"
    env = IntegratorEnv(num_envs=4)
    direct = sample(env, steps=3, seed=5)
    viaenv = StableEnv.sample(env, steps=3, seed=5)
    assert torch.equal(direct["states"], viaenv["states"])
