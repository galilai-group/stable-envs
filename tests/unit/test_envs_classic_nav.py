"""Classic-control and navigation State and renderer contracts.

Classic-control cells step Gymnasium on CPU with full lifecycle/RNG State; exact SWM parity is checked
in test_native_classic_semantics.py. TwoRoom, Piecewise and SimplePointMaze have native dynamics with
task State; TwoRoomTarget and PiecewiseZones have structural coverage only.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.stepping import branch_step, branch_steps

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch
from tensordict import TensorDict

import stable_envs as snv

pytestmark = pytest.mark.unit

CLASSIC = ("CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous")
NAV = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones", "SimplePointMaze")
ALL = CLASSIC + NAV
NAV_TASK = ("TwoRoom", "Piecewise", "SimplePointMaze")
PURE = NAV

#: Each cell's public State keys, written out so a change of the public vocabulary fails.
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


RNG_KEYS = [("rng", name) for name in ("state", "increment", "has_uint32", "uinteger")]
FULL_KEYS = {
    **KEYS,
    "TwoRoom": KEYS["TwoRoom"] + [("target", "position")],
    "Piecewise": KEYS["Piecewise"] + [("target", "position")],
    "SimplePointMaze": KEYS["SimplePointMaze"]
    + [("goal", "position"), ("task", "cached_goal_agent_position"), ("task", "state_is_float64")],
    "Acrobot": KEYS["Acrobot"] + [("lifecycle", "elapsed_steps")] + RNG_KEYS,
    "MountainCar": KEYS["MountainCar"] + [("lifecycle", "elapsed_steps")] + RNG_KEYS,
    "MountainCarContinuous": KEYS["MountainCarContinuous"]
    + [("task", "state_is_float32"), ("lifecycle", "elapsed_steps")]
    + RNG_KEYS,
    "CartPole": KEYS["CartPole"]
    + [
        ("lifecycle", "elapsed_steps"),
        ("task", "steps_beyond_terminated"),
        ("task", "has_steps_beyond_terminated"),
    ]
    + RNG_KEYS,
    "Pendulum": KEYS["Pendulum"]
    + [
        ("task", "has_last_u"),
        ("lifecycle", "elapsed_steps"),
    ]
    + RNG_KEYS,
}
PHYSICS = {
    "CartPole": ("gravity", "masscart", "masspole", "length", "force_mag", "tau"),
    "Pendulum": ("g", "m", "l", "max_torque", "max_speed", "dt"),
    "Acrobot": (
        "link_length_1",
        "link_length_2",
        "link_mass_1",
        "link_mass_2",
        "link_com_pos_1",
        "link_com_pos_2",
        "link_moi",
    ),
    "MountainCar": ("gravity", "force", "max_speed"),
    "MountainCarContinuous": ("power", "max_speed"),
}
for _name, _leaves in PHYSICS.items():
    FULL_KEYS[_name] = FULL_KEYS[_name] + [("physics", leaf) for leaf in _leaves]


def _initialize(env, name, seed):
    if name in CLASSIC + NAV_TASK:
        env.reset(seed=seed)
    else:
        env._set_render_state(env.sample_states(env.B, np.random.default_rng(seed)))


def _env(name, n=4):
    return getattr(snv, name)(num_envs=n, device="cpu")


def _actions(env, n, steps=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (n, env.action_dim) if steps is None else (steps, n, env.action_dim)
    if env.action_spec.discrete_n is not None and type(env).__name__ in CLASSIC:
        return torch.randint(0, env.action_spec.discrete_n, shape, generator=g, dtype=env.action_spec.dtype)
    lo, hi = float(env.action_low), float(env.action_high)
    return lo + (hi - lo) * torch.rand(*shape, generator=g)


@pytest.mark.parametrize("name", ALL)
def test_every_cell_builds_with_num_envs_and_device_and_closes_idempotently(name):
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
    assert [k for k, _, _ in layout] == FULL_KEYS[name]
    assert sum(w for _, w, o in layout if o is not None) == env.state_dim
    rows = torch.as_tensor(env.sample_states(4, np.random.default_rng(0)), device=env.device)
    td = env._unpack_render_state(rows)
    assert isinstance(td, TensorDict) and list(td.batch_size) == [4]
    assert torch.equal(env._pack_render_state(td), rows), "pack/unpack must round-trip the frozen packed layout"
    env.close()


@pytest.mark.parametrize("name", CLASSIC)
def test_classic_velocities_are_full_state_and_absent_from_render_projection(name):
    with _env(name) as env:
        state = env.reset(seed=17)
        hidden = {key for key, _, offset in env._layout() if offset is None}
        velocities = {(field,) for i, field in enumerate(env.sim_state_names) if i not in set(env.render_from_sim)}
        assert velocities and velocities <= hidden
        projection = env._unpack_render_state(env._pack_render_state(state))
        for key in velocities:
            assert key in state.keys(True, True)
            assert key not in projection.keys(True, True)
        with pytest.raises(ValueError, match="missing"):
            env.set_state(projection)


@pytest.mark.parametrize("name", CLASSIC)
def test_classic_physical_state_is_lossless_and_projects_to_float32(name):
    with _env(name) as env:
        state = env.reset(seed=17)
        assert all(state[key].dtype == torch.float64 for key in KEYS[name])
        rows = env._pack_render_state(state)
        assert rows.dtype == torch.float32
        assert torch.equal(rows, env._get_render_state())


@pytest.mark.parametrize("name", PURE)
def test_snapshot_branch_is_bit_identical_to_the_stateful_step(name):
    env = _env(name)
    s0 = env.sample_states(4, np.random.default_rng(0))
    if name in NAV_TASK:
        env.reset(seed=17)
    else:
        env._set_render_state(s0)
    a = _actions(env, 4)

    functional = branch_step(env, env.state, a)
    stateful = env.step(a, frames=None)["state"]
    assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    for k in FULL_KEYS[name]:
        assert torch.equal(functional[k], stateful[k]), f"{'.'.join(k)} differs"
    env.close()


@pytest.mark.parametrize("name", PURE)
def test_snapshot_branch_does_not_touch_the_env_and_unroll_is_repeated_transition(name):
    env = _env(name)
    s0 = env.sample_states(4, np.random.default_rng(1))
    if name in NAV_TASK:
        env.reset(seed=17)
    else:
        env._set_render_state(s0)
    held = env._get_render_state().clone()
    acts = _actions(env, 4, steps=4, seed=1)

    traj = branch_steps(env, env.state, acts)
    assert torch.equal(env._get_render_state(), held), "test branching must restore the original simulator"
    assert list(traj.batch_size) == [4, 4]

    cur = env.state
    for t in range(4):
        cur = branch_step(env, cur, acts[t])
        for k in FULL_KEYS[name]:
            assert torch.equal(traj[t][k], cur[k])

    # Four stateful steps land on the same numbers.
    for t in range(4):
        env.step(acts[t], frames=None)
    for k in FULL_KEYS[name]:
        assert torch.equal(traj[3][k], env.state[k]), f"{'.'.join(k)} drifted over four steps"
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_render_takes_a_state_a_packed_row_or_nothing_and_draws_the_same_frame(name):
    env = _env(name, 2)
    _initialize(env, name, 2)
    state = env.get_state()
    rows = env._pack_render_state(state)
    env._set_render_state(rows)
    a = env._render()
    b = env._render(rows)
    c = env.render(state)
    assert a.dtype == torch.uint8 and a.shape[0] == 2
    assert torch.equal(a, b) and torch.equal(a, c)
    assert tuple(env.frame_spec.shape) == tuple(a.shape[1:])
    env.close()


@pytest.mark.parametrize("name", ("TwoRoomTarget", "PiecewiseZones"))
def test_unrecovered_step_reports_only_its_implemented_state(name):
    env = _env(name, 2)
    env._set_render_state(env.sample_states(2, np.random.default_rng(3)))
    out = env.step(_actions(env, 2), frames=None)
    assert set(out.keys()) == {"state", "states", "valid", "state_valid"}, (
        "these cells have no reward or termination semantics, so those keys must be absent"
    )
    assert env.has_dynamics


@pytest.mark.parametrize("name", ALL)
def test_the_registered_default_frame_size_and_the_specs_agree(name):
    env = _env(name, 2)
    assert env.size == env.default_size
    assert tuple(env.action_spec.shape) == (env.action_dim,)
    spec = env.state_spec
    assert sorted(tuple(k) if isinstance(k, tuple) else (k,) for k in spec.keys(nested=True)) == sorted(
        FULL_KEYS[name]
    )
    env.close()


def test_the_swm_reference_bridge_uses_the_source_default_goal_visibility():
    """Canonical reference rows contain agent and reward-goal coordinates."""

    class _Leaf:
        _value = None

        @property
        def value(self):
            return self._value

    calls = []

    class _Stub:
        state = np.zeros(2, np.float32)
        variation_space = {
            "agent": {"position": _Leaf()},
            "goal": {"position": _Leaf()},
        }

        def reset(self, seed=None, options=None):
            calls.append(("reset", seed, options))

        def render(self, *args):
            calls.append(("render", args))
            return np.zeros((500, 500, 3), np.uint8)

        @property
        def unwrapped(self):
            return self

    ref = _Stub()
    want = np.array([[1.0, 2.0, 4.5, 4.5], [3.0, 4.0, 2.5, 3.5]], np.float32)
    frames = snv.SimplePointMaze.reference_render(ref, want)

    assert frames.shape == (2, 500, 500, 3)
    assert calls[0] == ("reset", 0, {"variation": []})
    assert [c for c in calls if c[0] == "render"] == [("render", ("rgb_array",))] * 2
    assert ref.state.dtype == np.float32 and np.array_equal(ref.state, want[1, :2])
    assert np.array_equal(ref.variation_space["goal"]["position"]._value, want[1, 2:])
    assert snv.SimplePointMaze.reference_kwargs == {"show_goal": True}


def test_simple_point_maze_goal_is_an_agg_alpha_composite():
    """The source's alpha=0.5 goal disc can never show a requested RGB literally."""
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    decl = snv.SimplePointMaze.color_exactness_applicability
    assert decl["leaves"] == ("goal.color",) and decl["max_shading_factor"] == 128 / 255
    bg = np.array([255, 255, 255])
    for target in ([17, 33, 200], [200, 17, 33]):
        fig = Figure(figsize=(1, 1))
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        fig.patch.set_facecolor(bg / 255.0)
        ax.add_patch(Circle((0.5, 0.5), 0.4, facecolor=np.asarray(target) / 255.0, alpha=0.5))
        fig.canvas.draw()
        centre = np.asarray(fig.canvas.buffer_rgba())[50, 50, :3].astype(int)
        assert centre.tolist() == np.floor(bg + (np.asarray(target) - bg) * 128 / 255).astype(int).tolist()
        assert centre.tolist() != target


def test_a_classic_cell_whose_layout_does_not_cover_its_packed_row_fails_at_import():
    """The check that lets ``_Classic.state`` skip the base class's per-call hidden-field guard."""
    from stable_envs.envs.classic_control.envs import _Classic

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


@pytest.mark.parametrize("name", PURE)
def test_snapshot_branch_still_equals_step_after_the_env_has_been_stepped(name):
    """Branching equals stepping mid-trajectory, not only straight after ``set_state``.

    A freshly loaded float32 state hides precision loss from rebuilding the state off the packed row.
    """
    env = _env(name)
    env._set_render_state(env.sample_states(4, np.random.default_rng(5)))
    for _ in range(5):
        env.step(_actions(env, 4, seed=5), frames=None)

    a = _actions(env, 4, seed=6)
    functional = branch_step(env, env.state, a)
    stateful = env.step(a, frames=None)["state"]
    assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    for k in FULL_KEYS[name]:
        assert torch.equal(functional[k], stateful[k]), f"{'.'.join(k)} differs after five steps"
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_a_step_result_is_not_a_view_of_the_env_that_produced_it(name):
    """A held ``out["state"]`` does not change when the env steps again."""
    env = _env(name)
    _initialize(env, name, 7)
    out = env.step(_actions(env, 4, seed=7), frames=None)
    held = {k: out["state"][k].clone() for k in FULL_KEYS[name]}
    env.step(_actions(env, 4, seed=8), frames=None)
    for k in FULL_KEYS[name]:
        assert torch.equal(out["state"][k], held[k]), (
            f"{'.'.join(k)} in a StepResult changed when the env was stepped again"
        )
    env.close()


@pytest.mark.parametrize("name", ALL)
def test_step_returns_exactly_env_state(name):
    """``step``'s state equals ``env.state`` after several steps.

    Classic control builds the result from the simulator buffer, while ``env.state`` round-trips the
    packed row; both must give the same numbers.
    """
    env = _env(name)
    _initialize(env, name, 9)
    for _ in range(3):
        env.step(_actions(env, 4, seed=9), frames=None)
    out = env.step(_actions(env, 4, seed=10), frames=None)
    held = env.state
    assert sorted(out["state"].keys()) == sorted(held.keys())
    for k in FULL_KEYS[name]:
        assert torch.equal(out["state"][k], held[k]), k
        assert out["state"][k].dtype == held[k].dtype
    env.close()


@pytest.mark.parametrize("name", CLASSIC)
def test_recovered_classic_reports_authoritative_outputs_and_typed_state(name):
    with _env(name, 2) as env:
        result = env.step(_actions(env, 2), frames=None)
        assert {
            "state",
            "reward",
            "terminated",
            "truncated",
            "observation",
            "info_state",
            "states",
            "valid",
            "state_valid",
        } == set(result.keys())
        assert result["reward"].dtype == torch.float64
        assert result["terminated"].dtype == result["truncated"].dtype == torch.bool
        assert env.execution_mode == "DELEGATED_CPU"
        assert set(result["state"].keys(True, True)) == {key[0] if len(key) == 1 else key for key in FULL_KEYS[name]}
        for key in result["state"].keys(True, True):
            assert result["state"][key].dtype == env.state_spec[key].dtype


def test_piecewise_draws_its_hidden_target_only_from_full_state():
    with _env("Piecewise", 2) as env:
        state = env.reset(seed=0)
        plain = env.render(state)
        env.set_variation({"rendering.render_target": [1]})
        assert not torch.equal(env.render(state), plain)
        with pytest.raises(ValueError, match="full State"):
            env._render(env._pack_render_state(state).clone())


@pytest.mark.parametrize("name", ("CartPole", "Acrobot"))
def test_drawn_physics_lengths_render_as_the_official_frame(name):
    """Per-lane pole and link lengths, including every Acrobot joint radius the range reaches (7-20)."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("pygame")
    if name == "CartPole":
        lengths = {"physics.length": np.asarray([0.25, 0.4, 0.6, 0.75], np.float32)}
    else:
        total = np.asarray([min(25 / (r + 0.5) - 0.2, 2.98) for r in range(7, 21)])
        lengths = {f"physics.link_length_{i}": (total / 2).astype(np.float32) for i in (1, 2)}
    n = len(next(iter(lengths.values())))
    with _env(name, 1) as env:
        states = env.sample_states(n, np.random.default_rng(5))
        ours = env._render(torch.as_tensor(states), variation={k: v.reshape(n, 1) for k, v in lengths.items()})
        ref = env.reference_env()
        ref.reset(seed=0)
        radii = set()
        for i in range(n):
            if name == "CartPole":
                ref.length = float(lengths["physics.length"][i])
            else:
                ref.LINK_LENGTH_1 = float(lengths["physics.link_length_1"][i])
                ref.LINK_LENGTH_2 = float(lengths["physics.link_length_2"][i])
                radii.add(int(0.1 * 500 / ((ref.LINK_LENGTH_1 + ref.LINK_LENGTH_2 + 0.2) * 2)))
            env.reference_set_state(ref, states[i])
            assert np.array_equal(ours[i].numpy(), ref.render()), (name, i)
        ref.close()
    assert name == "CartPole" or radii == set(range(7, 21))
