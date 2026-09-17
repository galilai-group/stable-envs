"""The State: what ``pack``/``unpack`` promise, what ``reset`` promises, and what the specs report.

The load-bearing claim of Step 9's state work is that NOTHING MOVED. The entity-grouped TensorDict is
a view of the same five floats a PushT corpus has always stored, in the same order, so every frozen
digest and every certification check still sees the Step-8 row. These tests assert that as an identity
rather than as an intention.

CPU only; PushT needs no reference package to be constructed and rendered.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs.specs import CompositeSpec, Spec

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def env():
    e = snv.PushT(num_envs=8, device="cpu")
    yield e
    e.close()


def test_the_state_keys_are_the_entities_and_not_the_packed_row(env):
    st = env.reset(seed=0)
    assert set(st.keys()) == {"agent", "block"}
    assert set(st["agent"].keys()) == {"position", "velocity"}
    assert set(st["block"].keys()) == {"position", "angle"}
    assert st["agent", "position"].shape == (8, 2)
    assert st["block", "angle"].shape == (8, 1)
    assert list(st.batch_size) == [8]


def test_the_packed_row_is_the_step8_layout_unchanged(env):
    assert env.state_dim == 5
    assert env.state_names == ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
    assert env.state_names_of(("agent", "position")) == ["agent_x", "agent_y"]
    assert env.state_names_of(("block", "angle")) == ["block_angle"]
    # the hidden dynamical field occupies no packed column at all
    assert env.state_names_of(("agent", "velocity")) == []


def test_pack_unpack_round_trips_a_corpus_row_for_row(env):
    rows = torch.as_tensor(env.sample_states(8, np.random.default_rng(3)), dtype=torch.float32)
    back = env.pack(env.unpack(rows))
    assert torch.equal(rows, back), "pack(unpack(rows)) must be the identity on the packed layout"


def test_unpack_fills_the_hidden_field_with_the_condition_set_state_produces(env):
    rows = torch.as_tensor(env.sample_states(8, np.random.default_rng(4)), dtype=torch.float32)
    st = env.unpack(rows)
    assert float(st["agent", "velocity"].abs().max()) == 0.0
    env.set_state(rows)
    assert float(env.state["agent", "velocity"].abs().max()) == 0.0


def test_pack_of_the_live_state_is_exactly_get_state(env):
    env.reset(seed=1)
    assert torch.equal(env.pack(env.state), env.get_state())


def test_render_accepts_a_state_and_packed_rows_and_draws_the_same_pixels(env):
    st = env.reset(seed=2)
    assert torch.equal(env.render(st), env.render(env.pack(st)))
    assert torch.equal(env.render(st), env.render())


def test_a_layout_that_does_not_cover_the_packed_row_is_refused():
    class Broken(snv.PushT):
        state_layout = ((("agent", "position"), 2, 0),)

    e = Broken(num_envs=2, device="cpu")
    with pytest.raises(ValueError, match="covers 2 of the 5"):
        e.pack(e.unpack(torch.zeros(2, 5)))
    e.close()


def test_reset_is_deterministic_in_its_seed(env):
    a = env.pack(env.reset(seed=7)).clone()
    b = env.pack(env.reset(seed=7)).clone()
    c = env.pack(env.reset(seed=8)).clone()
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_reset_draws_from_the_envs_own_distribution(env):
    st = env.reset(seed=0)
    ap = st["agent", "position"]
    assert float(ap.min()) >= 50.0 and float(ap.max()) <= 450.0


def test_the_specs_describe_the_state_the_actions_the_frames_and_the_style(env):
    ss = env.state_spec
    assert isinstance(ss, CompositeSpec)
    assert ss.keys(nested=True) == [("agent", "position"), ("agent", "velocity"),
                                    ("block", "position"), ("block", "angle")]
    assert ss.contains(env.reset(seed=0))

    a = env.action_spec
    assert isinstance(a, Spec) and a.shape == (2,)
    assert float(np.min(np.asarray(a.low))) == 0.0 and float(np.max(np.asarray(a.high))) == 512.0

    o = env.observation_spec
    assert o.shape == (env.size, env.size, 3) and o.dtype is torch.uint8

    v = env.variation_spec
    assert v.shape == (env.style_dim,)
    # one name per COLUMN: a width-3 colour leaf contributes three of them, and the set of paths is
    # still exactly ``style_names``.
    assert len(v.names) == env.style_dim
    assert sorted({n.split("[")[0] for n in v.names}) == sorted(env.style_names)
    assert v.contains(torch.as_tensor(env.default_style()).reshape(1, -1))


def test_num_envs_is_the_batch_and_len_agrees(env):
    assert env.num_envs == 8 == len(env) == env.B
