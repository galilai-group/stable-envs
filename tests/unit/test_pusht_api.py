"""Snapshot branching and state ownership for the existing PushT implementation.

These checks do not establish upstream contact-history equivalence or promote PushT.
"""

from __future__ import annotations

import pytest

from tests.helpers.stepping import branch_step, branch_steps

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv

pytestmark = pytest.mark.unit


@pytest.fixture()
def env():
    e = snv.PushT(num_envs=6, device="cpu")
    yield e
    e.close()


def _actions(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 60.0 + 392.0 * torch.rand(n, 2, generator=g)


def test_step_is_stateful_and_returns_a_stepresult_carrying_the_state(env):
    env.reset(seed=0)
    before = env._get_render_state().clone()
    out = env.step(_actions(6), capture="states")
    assert set(out.keys()) == {"state"}, (
        "PushT has no reward and no termination in stable-envs and must omit them rather than fabricate zeros"
    )
    assert list(out.batch_size) == [6]
    assert torch.equal(env._pack_render_state(out["state"]), env._get_render_state())
    assert not torch.equal(before, env._get_render_state()), "step must advance the env's held state"


def test_snapshot_branch_is_pure_in_the_state_and_leaves_the_env_alone(env):
    st = env.reset(seed=1)
    held = env._get_render_state().clone()
    a = _actions(6, 1)
    nxt = branch_step(env, st, a)
    assert torch.equal(env._get_render_state(), held), "transition must not touch the env's buffers"
    again = branch_step(env, st, a)
    assert torch.equal(env._pack_render_state(nxt), env._pack_render_state(again))
    assert torch.equal(nxt["agent", "velocity"], again["agent", "velocity"])


def test_snapshot_branch_equals_step_to_the_bit_on_the_same_state(env):
    st = env.reset(seed=2)
    a = _actions(6, 2)
    functional = branch_step(env, st, a)
    stateful = env.step(a, capture="states")["state"]
    assert torch.equal(env._pack_render_state(functional), env._pack_render_state(stateful))
    assert torch.equal(functional["agent", "velocity"], stateful["agent", "velocity"])


def test_the_velocity_is_real_state_and_carrying_it_is_what_makes_transition_pure(env):
    """Two transitions from states that differ ONLY in the hidden field must differ."""
    st = env.reset(seed=3)
    moving = st.clone()
    moving["agent", "velocity"] = torch.full_like(moving["agent", "velocity"], 25.0)
    a = _actions(6, 3)
    assert not torch.equal(env._pack_render_state(branch_step(env, st, a)), env._pack_render_state(branch_step(env, moving, a)))


def test_snapshot_branch_sequence_matches_repeated_steps(env):
    st = env.reset(seed=4)
    acts = torch.stack([_actions(6, 10 + t) for t in range(5)])
    traj = branch_steps(env, st, acts)
    assert list(traj.batch_size) == [5, 6]

    cur = st
    for t in range(5):
        cur = branch_step(env, cur, acts[t])
        assert torch.equal(env._pack_render_state(traj[t]), env._pack_render_state(cur))
        assert torch.equal(traj[t]["agent", "velocity"], cur["agent", "velocity"])
    assert torch.equal(env._get_render_state(), env._pack_render_state(st)), "unroll must not advance the env"


def test_branch_reference_refuses_actions_without_a_time_axis(env):
    st = env.reset(seed=5)
    with pytest.raises(ValueError, match="time first"):
        branch_steps(env, st, _actions(6))


def test_a_cell_without_dynamics_says_so_precisely():
    """Renderer-only cells are the majority and the refusal is a design statement, not a gap."""

    class Renderless(snv.PushT):
        _step = snv.Env._step

    e = Renderless(num_envs=2, device="cpu")
    assert not e.has_dynamics
    for call in (lambda: e.step(torch.zeros(2, 2), capture="states"), lambda: branch_step(e, e.reset(seed=0), torch.zeros(2, 2))):
        with pytest.raises(NotImplementedError) as exc:
            call()
        assert "full simulator semantics" in str(exc.value) and "Renderless" in str(exc.value)
    # reset and render still work, which is the other half of the statement
    assert e._render(e.sample_states(2)).shape[0] == 2
    e.close()


def test_the_step_result_is_built_by_the_shared_helper(env):
    """One helper, so forty cells cannot each get the batch size or the device subtly different."""
    env.reset(seed=0)
    out = env.step(_actions(6), capture="states")
    assert isinstance(out, snv.StepResult)
    assert list(out.batch_size) == [6] and out.device == env.device

    built = env._step_result(env.state, reward=torch.zeros(6), done=torch.zeros(6, dtype=torch.bool))
    assert set(built.keys()) == {"state", "reward", "done"}
    assert list(built["reward"].shape) == [6] and list(built["done"].shape) == [6]
    assert set(env._step_result(env.state, reward=None).keys()) == {"state"}, (
        "a cell with no reward semantics passes None and the key is ABSENT, not zero"
    )


def test_a_cell_with_hidden_state_and_no_hook_fails_loudly_instead_of_reporting_zeros():
    """The one template hazard worth a test: forgetting the hook must not be a silent wrong answer.

    A hidden field left unfilled would come back as zeros — ``env.state`` looking fine while
    ``transition(env.state, a)`` quietly stopped matching ``step(a)``.
    """

    class Forgetful(snv.PushT):
        _hidden_state = snv.Env._hidden_state

    e = Forgetful(num_envs=2, device="cpu")
    e._set_render_state(e.sample_states(2))
    with pytest.raises(NotImplementedError) as exc:
        e.state
    msg = str(exc.value)
    assert "agent.velocity" in msg and "offset " in msg and "ZEROS" in msg
    assert "_hidden_state()" in msg, "the message must name the hook that fixes it"
    e.close()


def test_the_hook_reports_the_live_velocity_and_env_state_round_trips_through_step(env):
    env.reset(seed=6)
    assert float(env.state["agent", "velocity"].abs().max()) == 0.0
    env.step(_actions(6, 6), capture="states")
    live = env.state["agent", "velocity"]
    assert float(live.abs().max()) > 0.0, "after a step the agent is moving and state must say so"
    assert torch.equal(live, env._hidden_state()[("agent", "velocity")])
    # the whole point: state -> transition must reproduce the next step exactly
    a = _actions(6, 7)
    assert torch.equal(env._pack_render_state(branch_step(env, env.state, a)), env._pack_render_state(env.step(a, capture="states")["state"]))


def test_pusht_declares_that_it_has_dynamics(env):
    assert env.has_dynamics and snv.info("PushT")["has_dynamics"]
