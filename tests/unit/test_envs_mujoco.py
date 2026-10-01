"""MuJoCo-wrapper families on the Env contract: the constructor, the State, the refusal.

The oracle that the BEHAVIOR did not change is the certified MuJoCo-wrapper corpus (both trees,
byte-identical frames over 24 cells). This file pins the claims that are new on the public API and therefore
have no baseline to compare against, across the seventeen families these families cover:

  * **the constructor**, checked from the SIGNATURE for every registered class of these families and
    not only for the ones this host can build. Most of these families need an optional dependency
    (``minigrid``, ``gym-sokoban``, ``gymnasium-robotics``, ``procgen``, ``jaxmarl``, ``pgx``,
    ``griddly``, Box2D, ``mujoco_warp``, nvdiffrast), so a test that instantiated would SKIP on the
    machine where the question is asked and the property would quietly stop being enforced. This is
    the same argument ``test_public_api.py`` makes for its own ratchet, applied per family;
  * **``state_layout`` is a real layout**: every declared field is a key of the public State, the
    offsets tile the packed row exactly, and ``pack``/``unpack`` round-trip it — checked on the
    class where the layout is a class attribute, and on an instance where it comes from a compiled
    model;
  * **the renderer-only refusal** (contract §5): a family with no batched simulator raises
    ``NotImplementedError`` naming itself from BOTH ``step`` and ``transition``, while ``reset`` and
    ``render`` keep working;
  * **``close``**: idempotent, a closed env refuses to render, the context manager closes.

Only the cells this interpreter can actually construct are exercised for the last three; the
constructor property is enforced for all of them. Sizes are small (2-4 envs).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.helpers.stepping import branch_step, branch_steps

pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch

from stable_envs import _dynamics as D
from stable_envs.registry import REGISTRY

pytestmark = pytest.mark.unit

#: The seventeen modules these families cover, by the last component of their import path.
FAMILIES = (
    "mujoco_locomotion",
    "humanoidbench",
    "metaworld",
    "robocasa",
    "maze2d",
    "carracing",
    "crafter",
    "griddly",
    "highway",
    "jumanji",
    "lunarlander",
    "minigrid",
    "minihack",
    "overcooked",
    "pgx",
    "procgen",
    "sokoban",
)

#: The two cells in these families that ship a batched simulator. Everything else is a renderer over
#: someone else's physics and says so by raising.
WITH_DYNAMICS = ("MiniGridTopDown", "Sokoban")


def _owned_classes() -> dict:
    """``{registered name: class}`` for every class of the families these families cover.

    Read off the registry rather than by importing each module, so a family whose optional
    dependency is missing simply contributes nothing instead of erroring — and so a class that is
    added to one of these modules tomorrow is covered without editing a list here.
    """
    return {n: c for n, c in REGISTRY.items() if getattr(c, "__module__", "").rsplit(".", 1)[-1] in FAMILIES}


def _constructible(name: str, cls, **kw):
    """One CPU env, or a skip naming the dependency this host does not have."""
    try:
        return cls(num_envs=kw.pop("num_envs", 2), device="cpu", **kw)
    except Exception as e:  # noqa: BLE001 — the reason is the skip message
        if isinstance(e, (ImportError, ModuleNotFoundError, OSError, FileNotFoundError)):
            pytest.skip(f"{name} is not constructible on this host: {type(e).__name__}: {e}")
        raise


OWNED = sorted(_owned_classes())


def test_this_workstream_owns_some_registered_classes():
    """A guard on the guard: if discovery silently stopped, every test below would vacuously pass."""
    assert OWNED, (
        "no registered class resolves to a MuJoCo-wrapper family; the tests below "
        "parametrise over this list and would all pass on an empty one"
    )


@pytest.mark.parametrize("name", OWNED)
def test_every_owned_class_takes_the_step9_constructor(name):
    """``num_envs=`` reaches the class, from the SIGNATURE — no construction, so no optional dep."""
    params = inspect.signature(REGISTRY[name].__init__).parameters
    assert (
        "num_envs" in params or "kw" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    ), f"{name} does not accept num_envs=; its signature is {params}"
    assert "batch" not in params, f"{name} declares batch= as a constructor spelling"


@pytest.mark.parametrize("name", OWNED)
def test_the_declared_state_layout_tiles_the_packed_row(name):
    """Every packed column belongs to exactly one declared field, and the keys are real State keys.

    ``Env._check_layout`` raises when the offsets do not cover ``state_dim``; asking for
    ``env.state`` is what exercises it, together with ``_hidden_state`` for any field the row does
    not carry. ``pack(env.state)`` must then be ``get_state()`` to the BIT: the public State is a
    view of the packed row and not a second, rounded copy of it.
    """
    env = _constructible(name, REGISTRY[name])
    try:
        env._set_render_state(env.sample_states(env.num_envs, np.random.default_rng(0)))
        layout = env._layout()
        state = env.state
        assert sum(w for _, w, off in layout if off is not None) == env.state_dim
        for key, w, _ in layout:
            assert tuple(state[key].shape) == (env.num_envs, w), f"{name}: {key}"
        assert torch.equal(env._pack_render_state(state), env._get_render_state())
        assert torch.equal(env._pack_render_state(env._unpack_render_state(env._get_render_state())), env._get_render_state())
    finally:
        env.close()


@pytest.mark.parametrize("name", OWNED)
def test_render_takes_a_state_a_packed_row_or_nothing(name):
    """The three spellings are the same frame, and ``frame_spec`` describes it."""
    env = _constructible(name, REGISTRY[name])
    try:
        rows = env.sample_states(env.num_envs, np.random.default_rng(0))
        env._set_render_state(rows)
        px = env._render()
        assert torch.equal(px, env._render(env.state))
        assert torch.equal(px, env._render(env._get_render_state()))
        assert px.dtype is torch.uint8 and px.shape[0] == env.num_envs
        assert tuple(env.frame_spec.shape) == tuple(px.shape[1:])
    finally:
        env.close()


@pytest.mark.parametrize("name", OWNED)
def test_a_cell_without_a_simulator_refuses_step_and_transition_by_name(name):
    """Contract §5: the refusal says what is missing and what to do, and it is not an AttributeError.

    ``reset`` and ``render`` keep working — that is the point of the refusal being explicit rather
    than the method being absent. The two cells that DO ship a simulator are checked in the opposite
    direction: they must not raise.
    """
    cls = REGISTRY[name]
    steppable = any(b.__name__ in WITH_DYNAMICS for b in cls.__mro__)
    env = _constructible(name, cls)
    try:
        env._set_render_state(env.sample_states(env.num_envs, np.random.default_rng(0)))
        if steppable:
            assert D.has_dynamics(env)
            out = env.step(torch.zeros(env.num_envs, env.action_dim), frames=None)
            assert list(out.keys()) == ["state"], (
                f"{name}: step fabricated {set(out.keys()) - {'state'}}; a cell with no reward or "
                f"termination semantics omits the key (contract §5)"
            )
            return
        assert not D.has_dynamics(env)
        for call in (
            lambda: env.step(torch.zeros(env.num_envs, 1), frames=None),
            lambda: branch_step(env, env.state, torch.zeros(env.num_envs, 1)),
        ):
            with pytest.raises(NotImplementedError) as e:
                call()
            assert type(env).__name__ in str(e.value) and "has dynamics" in str(e.value)
        assert env._render().shape[0] == env.num_envs  # render still works after the refusal
    finally:
        env.close()


@pytest.mark.parametrize("name", OWNED)
def test_close_is_idempotent_and_a_closed_env_refuses_to_render(name):
    """Resource lifetime is explicit (contract §5): never ``__del__``, and a closed env says so."""
    env = _constructible(name, REGISTRY[name])
    env.close()
    env.close()
    assert env.closed
    with pytest.raises(RuntimeError) as e:
        env._render()
    assert "closed" in str(e.value)
    with _constructible(name, REGISTRY[name]) as ctx:
        assert not ctx.closed
    assert ctx.closed


# --------------------------------------------------------------------------- the two steppable cells

STEPPABLE = [n for n in OWNED if any(b.__name__ in WITH_DYNAMICS for b in REGISTRY[n].__mro__)]


@pytest.mark.skipif(not STEPPABLE, reason="neither steppable family is constructible on this host")
@pytest.mark.parametrize("name", STEPPABLE)
def test_snapshot_branch_is_pure_and_bit_identical_to_the_stateful_step(name):
    """The whole claim of the functional path, and the one that can go wrong silently.

    ``transition`` must not touch the env, and it must return exactly what ``step`` would have
    written. They are the same ``_advance_into`` over different buffers, so a difference here is a
    buffer the refactor forgot — MiniGrid's carried object is the one that would show.
    """
    cls = REGISTRY[name]
    env = _constructible(name, cls, num_envs=4)
    try:
        rows = env.sample_states(4, np.random.default_rng(0))
        env._set_render_state(rows)
        g = torch.Generator().manual_seed(0)
        lo, hi = float(env.action_low), float(env.action_high)
        a = lo + (hi - lo) * torch.rand(4, env.action_dim, generator=g)
        held = env._get_render_state().clone()
        functional = branch_step(env, env.state, a)
        assert torch.equal(env._get_render_state(), held), f"{name}: transition wrote the env's buffer"
        stateful = env.step(a, frames=None)["state"]
        keys = [tuple(k) for k, _, _ in env._layout()]
        for k in keys:
            assert torch.equal(functional[k], stateful[k]), f"{name}: {k} differs"
        returned = stateful.clone()
        env.step(a, frames=None)
        for k in keys:
            assert torch.equal(stateful[k], returned[k]), f"{name}: returned State aliases env buffers"
    finally:
        env.close()


@pytest.mark.skipif(not STEPPABLE, reason="neither steppable family is constructible on this host")
@pytest.mark.parametrize("name", STEPPABLE)
def test_snapshot_branch_sequence_matches_repeated_steps_and_repeated_step(name):
    """Open loop, time first, and the same answer three ways over four steps."""
    env = _constructible(name, REGISTRY[name], num_envs=4)
    try:
        rows = env.sample_states(4, np.random.default_rng(0))
        g = torch.Generator().manual_seed(1)
        lo, hi = float(env.action_low), float(env.action_high)
        A = lo + (hi - lo) * torch.rand(4, 4, env.action_dim, generator=g)
        keys = [tuple(k) for k, _, _ in env._layout()]

        env._set_render_state(rows)
        traj = branch_steps(env, env.state, A)
        assert tuple(traj.batch_size) == (4, 4)

        env._set_render_state(rows)
        cur = env.state
        for t in range(4):
            cur = branch_step(env, cur, A[t])
            for k in keys:
                assert torch.equal(traj[t][k], cur[k]), f"{env.name}: step {t}, key {k}"

        env._set_render_state(rows)
        for t in range(4):
            env.step(A[t], frames=None)
        for k in keys:
            assert torch.equal(traj[3][k], env.state[k]), f"{env.name}: {k} after four steps"
    finally:
        env.close()


@pytest.mark.skipif("MiniGridEmpty" not in OWNED, reason="minigrid is not installed")
def test_minigrids_carried_object_is_hidden_state_the_packed_row_does_not_carry():
    """The one field in this workstream with offset ``None``, and the hazard it closes.

    ``unpack`` zeroes it, which is right for a state loaded from a corpus (``set_state`` empties the
    agent's hands, and so does the reference after ``reset``) and wrong for the env's own live state
    after a pickup. Without ``_hidden_state`` the env would report empty hands and
    ``transition(env.state, a)`` would stop matching ``step(a)`` — a wrong number rather than a
    failure. This asserts the declaration, the zero-fill and the splice.
    """
    env = _constructible("MiniGridDoorKey", REGISTRY["MiniGridDoorKey"], num_envs=4)
    try:
        layout = {tuple(k): (w, off) for k, w, off in env._layout()}
        assert layout[("agent", "carrying")] == (1, None)
        assert all(off is not None for k, (_, off) in layout.items() if k != ("agent", "carrying"))
        rows = env.sample_states(4, np.random.default_rng(0))
        env._set_render_state(rows)
        assert torch.all(env._unpack_render_state(env._get_render_state())["agent", "carrying"] == 0)
        # pickup is action 3; whatever it does to the four environments, env.state must report the
        # env's OWN carried codes and not the zeros unpack would fill in.
        env.step(torch.full((4, 1), 3.0), frames=None)
        assert torch.equal(env.state["agent", "carrying"], env._carry.reshape(-1, 1).to(torch.float32))
        # and the functional path must still agree with the stateful one MID-TRAJECTORY
        a = torch.full((4, 1), 2.0)
        functional = branch_step(env, env.state, a)
        stateful = env.step(a, frames=None)["state"]
        assert torch.equal(functional["agent", "carrying"], stateful["agent", "carrying"])
        assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    finally:
        env.close()
