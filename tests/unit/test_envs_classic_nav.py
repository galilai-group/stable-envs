"""Classic-control and navigation cells on the Env contract: the same pixels, a pure ``transition``.

The oracle that the BEHAVIOR did not change is
the certified classic/nav corpus (byte-identical
frames). This file pins the claims that are new on the public API and therefore have no baseline to compare
against, for all ten cells at once:

  * the ``num_envs=`` constructor and the lifecycle (``close``, the context manager);
  * ``state_layout``: what the public State's keys are, and — for classic control — that the
    velocities the packed row never carried are declared with offset ``None`` and answered by
    ``_hidden_state``;
  * ``pack``/``unpack`` round-trip the frozen packed layout, and ``pack(env.state)`` is exactly the
    packed row every renderer and digest is computed from;
  * ``transition`` equals the stateful ``step`` **to the bit**, and ``unroll`` equals repeated
    ``transition`` — for classic control over FOUR steps, because that is where a State that had
    rounded the dynamics to float32 would start to drift while a one-step check still passed;
  * ``render(State) == render(rows) == render()``;
  * the deprecated module paths still resolve to the merged module.

Sizes are small (4-8 envs) and every cell renders at its reference's native canvas, so the whole file
is a few seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch
from tensordict import TensorDict

import stable_envs as snv

pytestmark = pytest.mark.unit

CLASSIC = ("CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous")
NAV = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "SimplePointMaze")
ALL = CLASSIC + NAV

#: What each cell's public State is keyed by. Written out rather than derived from the class, so a
#: silent change of the public vocabulary is a failure and not a tautology.
KEYS = {
    "CartPole": [("x",), ("x_dot",), ("theta",), ("theta_dot",)],
    "Pendulum": [("theta",), ("theta_dot",), ("torque",)],
    "Acrobot": [("theta1",), ("theta2",), ("dtheta1",), ("dtheta2",)],
    "MountainCar": [("position",), ("velocity",)],
    "MountainCarContinuous": [("position",), ("velocity",)],
    "TwoRoom": [("agent", "position")],
    "TwoRoomTarget": [("agent", "position"), ("target", "position")],
    "Piecewise": [("agent", "position")],
    "PiecewiseZones": [("agent", "position")],
    "SimplePointMaze": [("agent", "position")],
}


def _env(name, n=4):
    return getattr(snv, name)(num_envs=n, device="cpu")


def _actions(env, n, steps=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (n, env.action_dim) if steps is None else (steps, n, env.action_dim)
    lo, hi = float(env.action_low), float(env.action_high)
    return lo + (hi - lo) * torch.rand(*shape, generator=g)


@pytest.mark.parametrize("name", ALL)
def test_every_cell_takes_the_step9_constructor(name):
    with _env(name, 3) as env:
        assert env.num_envs == 3 == env.B and len(env) == 3
        assert env.device.type == "cpu"
    assert env.closed
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env._render()


@pytest.mark.parametrize("name", ALL)
def test_the_state_keys_are_the_declared_ones_and_cover_the_packed_row(name):
    env = _env(name)
    layout = env._layout()
    assert [k for k, _, _ in layout] == KEYS[name]
    assert sum(w for _, w, o in layout if o is not None) == env.state_dim
    rows = torch.as_tensor(env.sample_states(4, np.random.default_rng(0)), device=env.device)
    td = env._unpack_render_state(rows)
    assert isinstance(td, TensorDict) and list(td.batch_size) == [4]
    assert torch.equal(env._pack_render_state(td), rows), "pack/unpack must round-trip the frozen packed layout"
    env.close()


@pytest.mark.parametrize("name", CLASSIC)
def test_classic_velocities_are_hidden_state_the_packed_row_does_not_carry(name):
    """The velocities are public state (so ``transition`` is pure) and no column of the row moved."""
    env = _env(name)
    hidden = [k for k, _, o in env._layout() if o is None]
    drawn = set(env.render_from_sim)
    assert hidden == [(nm,) for i, nm in enumerate(env.sim_state_names) if i not in drawn]
    assert hidden, "every classic-control cell has at least one velocity the frame never shows"
    # unpack fills them with zeros -- the condition set_state puts the env in, and the reference's
    # own condition after reset -- and the cell answers with the live value from its own buffer.
    rows = torch.as_tensor(env.sample_states(4, np.random.default_rng(0)), device=env.device)
    assert all(k not in env._unpack_render_state(rows).keys(True, True) for k in hidden)
    assert sorted(env._hidden_state()) == sorted(hidden)
    env.close()


@pytest.mark.parametrize("name", CLASSIC)
def test_the_classic_public_state_is_float64_and_packs_back_to_the_float32_row(name):
    """The dtype the family SIMULATES in, which is what makes the functional path the same simulator.

    A float32 State would still pass a one-step ``transition == step`` check (a corpus state is a
    float32 value with the velocities at rest) and drift from the second step on.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(0)))
    st = env.state
    assert all(st[k].dtype == torch.float64 for k in KEYS[name])
    assert torch.equal(env._pack_render_state(st), env._get_render_state())
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_transition_is_bit_identical_to_the_stateful_step(name):
    env = _env(name)
    s0 = env.sample_states(4, np.random.default_rng(0))
    env._set_render_state(s0)
    a = _actions(env, 4)

    functional = env.transition(env.state, a)
    stateful = env.step(a)["state"]
    assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    for k in KEYS[name]:
        assert torch.equal(functional[k], stateful[k]), f"{'.'.join(k)} differs"
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_transition_does_not_touch_the_env_and_unroll_is_repeated_transition(name):
    env = _env(name)
    s0 = env.sample_states(4, np.random.default_rng(1))
    env._set_render_state(s0)
    held = env._get_render_state().clone()
    acts = _actions(env, 4, steps=4, seed=1)

    traj = env.unroll(env.state, acts)
    assert torch.equal(env._get_render_state(), held), "transition/unroll must not write the env's buffers"
    assert list(traj.batch_size) == [4, 4]

    cur = env.state
    for t in range(4):
        cur = env.transition(cur, acts[t])
        for k in KEYS[name]:
            assert torch.equal(traj[t][k], cur[k])

    # ... and four steps of the stateful path land on exactly the same numbers.
    for t in range(4):
        env.step(acts[t])
    for k in KEYS[name]:
        assert torch.equal(traj[3][k], env.state[k]), f"{'.'.join(k)} drifted over four steps"
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_render_takes_a_state_a_packed_row_or_nothing_and_draws_the_same_frame(name):
    env = _env(name, 2)
    s0 = env.sample_states(2, np.random.default_rng(2))
    env._set_render_state(s0)
    a = env._render()
    b = env._render(torch.as_tensor(s0, device=env.device))
    c = env._render(env.state)
    assert a.dtype == torch.uint8 and a.shape[0] == 2
    assert torch.equal(a, b) and torch.equal(a, c)
    assert tuple(env.frame_spec.shape) == tuple(a.shape[1:])
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_step_reports_a_state_and_does_not_fabricate_a_reward(name):
    env = _env(name, 2)
    env._set_render_state(env.sample_states(2, np.random.default_rng(3)))
    out = env.step(_actions(env, 2))
    assert list(out.keys()) == ["state"], (
        "these cells have no reward and no termination semantics in stable-envs; an absent key is "
        "the honest answer and a zero would read as a measured one"
    )
    assert env.has_dynamics


def test_navigation_cells_live_in_the_merged_module():
    from stable_envs.envs import navigation

    for attr in (
        "IMG",
        "_ALPHA_FLOOR",
        "_alpha_blend",
        "_gaussian_dot",
        "DEFAULT_WALL_SHAPE",
        "DEFAULT_WALL_POS",
        "SimplePointMaze",
    ):
        assert hasattr(navigation, attr)
    assert {c.__module__ for c in (snv.TwoRoom, snv.Piecewise, snv.SimplePointMaze)} == {"stable_envs.envs.navigation"}


@pytest.mark.parametrize("name", ALL)
def test_the_registered_default_frame_size_and_the_specs_agree(name):
    env = _env(name, 2)
    assert env.size == env.default_size
    assert tuple(env.action_spec.shape) == (env.action_dim,)
    spec = env.state_spec
    assert sorted(tuple(k) if isinstance(k, tuple) else (k,) for k in spec.keys(nested=True)) == sorted(KEYS[name])
    env.close()


def test_the_shared_swm_reference_bridge_drives_simple_point_maze_the_old_way():
    """The one refactor this host cannot measure against the live reference, checked on a stub.

    ``SimplePointMaze``'s reference is a matplotlib figure, and neither local interpreter has
    matplotlib, so ``reference_render`` cannot be run against the official env here (the other four
    navigation cells share the identical ``_GymReferencePort`` gymnasium reference-env bridge and ARE
    measured against theirs — see findings §6). What is checkable without it is the call SEQUENCE, which is the whole of what
    the merge changed: the reset must pass ``options={"variation": []}`` (a plain reset resamples
    ``agent.position`` and every number measured afterwards would describe a different appearance),
    the state must be written to ``u.state`` as float32, and ``render`` must be called with
    ``"rgb_array"`` — matplotlib's env requires the mode, the torch-rendered cells refuse it.
    """
    calls = []

    class _Stub:
        state = np.zeros(2, np.float32)

        def reset(self, seed=None, options=None):
            calls.append(("reset", seed, options))

        def render(self, *args):
            calls.append(("render", args))
            return np.zeros((500, 500, 3), np.uint8)

        @property
        def unwrapped(self):
            return self

    ref = _Stub()
    want = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    frames = snv.SimplePointMaze.reference_render(ref, want)

    assert frames.shape == (2, 500, 500, 3)
    assert calls[0] == ("reset", 0, {"variation": []})
    assert [c for c in calls if c[0] == "render"] == [("render", ("rgb_array",))] * 2
    assert ref.state.dtype == np.float32 and np.array_equal(ref.state, want[1])
    assert snv.SimplePointMaze.reference_kwargs == {"show_goal": False}


@pytest.mark.parametrize("name", CLASSIC)
def test_env_state_is_the_simulator_buffer_and_not_the_widened_packed_row(name):
    """The two hooks together must cover the state exactly, with no field left to the lossy row.

    ``Env.state`` builds from ``unpack(get_state())`` and then takes ``_live_state()`` for the drawn
    fields and ``_hidden_state()`` for the hidden ones. This asserts the two answer for EVERY key —
    so no field is being served by the widened float32 row — and that what comes back is the
    simulator's own float64 buffer bit for bit, after the env has been stepped, which is when the row
    and the buffer have actually diverged.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(4)))
    for _ in range(3):
        env.step(_actions(env, 4))

    answered = {*env._live_state(), *env._hidden_state()}
    assert answered == set(KEYS[name]), "a field would be served by the lossy packed row"
    st = env.state
    for i, nm in enumerate(env.sim_state_names):
        assert torch.equal(st[nm], env.sim_state()[:, i : i + 1]), nm
    # and the row the renderer sees is still exactly the packed float32 one
    assert torch.equal(env._pack_render_state(st), env._get_render_state())
    env.close()


def test_a_classic_cell_whose_layout_does_not_cover_its_packed_row_fails_at_import():
    """The check that lets ``_Classic.state`` skip the base class's per-call hidden-field guard."""
    from stable_envs.envs.classic_control import _Classic

    with pytest.raises(TypeError, match="packed column"):

        class _Broken(_Classic):
            state_dim = 2
            sim_state_names = ("a", "b", "c")
            render_from_sim = (0,)

    with pytest.raises(TypeError, match="duplicate"):

        class _Dup(_Classic):
            state_dim = 2
            sim_state_names = ("a", "a", "c")
            render_from_sim = (0, 2)


@pytest.mark.parametrize("name", ALL)
def test_transition_still_equals_step_after_the_env_has_been_stepped(name):
    """The bit-equality must hold MID-TRAJECTORY, not only from a corpus state.

    A state loaded by ``set_state`` is float32 values with the velocities at rest, so a functional
    path that had lost precision — by rounding the state, or by rebuilding it from the packed row
    rather than from the buffer the dynamics write — still passes a check made straight after
    ``set_state``. Measured on CartPole, ``Env.state``'s drawn ``x`` differs from the simulator's own
    value by 9.6e-08 after five steps, which is exactly enough to break this. See
    ``_Classic.state``'s docstring.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(5)))
    for _ in range(5):
        env.step(_actions(env, 4, seed=5))

    a = _actions(env, 4, seed=6)
    functional = env.transition(env.state, a)
    stateful = env.step(a)["state"]
    assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    for k in KEYS[name]:
        assert torch.equal(functional[k], stateful[k]), f"{'.'.join(k)} differs after five steps"
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_a_step_result_is_not_a_view_of_the_env_that_produced_it(name):
    """``step`` hands back a state, not a window onto the env's buffer.

    The failure this pins is silent: a caller that keeps ``out["state"]`` and steps again would find
    the value it is holding had changed, with nothing raised and nothing logged. See findings §10,
    request R8 — the base class's ``unpack`` stopped cloning in foundation commit ``b1585fc``.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(7)))
    out = env.step(_actions(env, 4, seed=7))
    held = {k: out["state"][k].clone() for k in KEYS[name]}
    env.step(_actions(env, 4, seed=8))
    for k in KEYS[name]:
        assert torch.equal(out["state"][k], held[k]), (
            f"{'.'.join(k)} in a StepResult changed when the env was stepped again"
        )
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_step_returns_exactly_env_state(name):
    """Whatever ``step`` builds its result from, it must be what ``env.state`` reports.

    Classic control's ``step`` builds the StepResult from the simulator buffer directly instead of
    reading it back through the property, because the property round-trips the packed row to arrive
    where the buffer already is. That is only sound while the two are the same numbers, which is what
    this asserts — after several steps, when the row and the buffer have diverged.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(9)))
    for _ in range(3):
        env.step(_actions(env, 4, seed=9))
    out = env.step(_actions(env, 4, seed=10))
    held = env.state
    assert sorted(out["state"].keys()) == sorted(held.keys())
    for k in KEYS[name]:
        assert torch.equal(out["state"][k], held[k]), k
        assert out["state"][k].dtype == held[k].dtype
    env.close()
