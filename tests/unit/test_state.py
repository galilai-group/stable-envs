# Frozen compact-renderer/approximate-solver regressions. Public source-aligned
# PushT checkpoint semantics are covered by test_pusht_public_semantics.py.
"""The State: what ``pack``/``unpack`` promise, what ``reset`` promises, and what the specs report.

The load-bearing claim of the public State is that NOTHING MOVED. The entity-grouped TensorDict is
a view of the same five floats a PushT corpus has always stored, in the same order, so every frozen
digest and every certification check still sees the packed row. These tests assert that as an identity
rather than as an intention.

CPU only; PushT needs no reference package to be constructed and rendered.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs.envs.pusht.legacy import PushT as LegacyPushT
from stable_envs.specs import CompositeSpec, Spec

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def env():
    e = LegacyPushT(num_envs=8, device="cpu")
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
    assert env.state_dim == 5 == env.render_state_dim
    assert env.state_names == ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
    assert env.render_state_names == env.state_names
    assert env.state_names_of(("agent", "position")) == ["agent_x", "agent_y"]
    assert env.state_names_of(("block", "angle")) == ["block_angle"]
    # the hidden dynamical field occupies no packed column at all
    assert env.state_names_of(("agent", "velocity")) == []


def test_pack_unpack_round_trips_a_corpus_row_for_row(env):
    rows = torch.as_tensor(env.sample_states(8, np.random.default_rng(3)), dtype=torch.float32)
    back = env._pack_render_state(env._unpack_render_state(rows))
    assert torch.equal(rows, back), (
        "pack_render_state(unpack_render_state(rows)) must be the identity on the packed layout"
    )
    assert torch.equal(env._pack_render_state(env._unpack_render_state(rows)), back)


def test_private_unpack_omits_hidden_fields(env):
    rows = torch.as_tensor(env.sample_states(8, np.random.default_rng(4)), dtype=torch.float32)
    partial = env._unpack_render_state(rows)
    assert ("agent", "velocity") not in partial.keys(True, True)
    with pytest.raises(ValueError, match="missing agent.velocity"):
        env.set_state(partial)


def test_pack_of_the_live_state_is_exactly_get_state(env):
    env.reset(seed=1)
    assert torch.equal(env._pack_render_state(env.state), env._get_render_state())


def test_render_accepts_a_state_and_packed_rows_and_draws_the_same_pixels(env):
    st = env.reset(seed=2)
    assert torch.equal(env._render(st), env._render(env._pack_render_state(st)))
    assert torch.equal(env._render(st), env._render())


def test_a_layout_that_does_not_cover_the_packed_row_is_refused():
    class Broken(LegacyPushT):
        state_layout = ((("agent", "position"), 2, 0),)

    e = Broken(num_envs=2, device="cpu")
    with pytest.raises(ValueError, match="covers 2 of the 5"):
        e._pack_render_state(e._unpack_render_state(torch.zeros(2, 5)))
    e.close()


def test_reset_is_deterministic_in_its_seed(env):
    a = env._pack_render_state(env.reset(seed=7)).clone()
    b = env._pack_render_state(env.reset(seed=7)).clone()
    c = env._pack_render_state(env.reset(seed=8)).clone()
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_reset_draws_from_the_envs_own_distribution(env):
    st = env.reset(seed=0)
    ap = st["agent", "position"]
    assert float(ap.min()) >= 50.0 and float(ap.max()) <= 450.0


def test_the_specs_describe_the_state_the_actions_the_frames_and_the_style(env):
    ss = env.state_spec
    assert isinstance(ss, CompositeSpec)
    assert ss.keys(nested=True) == [
        ("agent", "position"),
        ("agent", "velocity"),
        ("block", "position"),
        ("block", "angle"),
    ]
    assert ss.contains(env.reset(seed=0))

    a = env.action_spec
    assert isinstance(a, Spec) and a.shape == (2,)
    assert float(np.min(np.asarray(a.low))) == 0.0 and float(np.max(np.asarray(a.high))) == 512.0

    o = env.frame_spec
    assert o.shape == (env.size, env.size, 3) and o.dtype is torch.uint8

    v = env.variation_spec
    assert v.shape == (env.variation_dim,)
    # one name per COLUMN: a width-3 colour leaf contributes three of them, and the set of paths is
    # still exactly ``variation_names``.
    assert len(v.names) == env.variation_dim
    assert sorted({n.split("[")[0] for n in v.names}) == sorted(env.variation_names)
    assert v.contains(torch.as_tensor(env.default_variation()).reshape(1, -1))


def test_num_envs_is_the_batch_and_len_agrees(env):
    assert env.num_envs == 8 == len(env) == env.B


def test_state_layout_entry_may_name_a_public_dtype():
    """A fourth ``state_layout`` element is the field's public dtype; pack still writes float32."""
    import torch

    from stable_envs.env import Env

    class F64(Env):
        state_dim = 3
        state_names = ("a", "b", "c")
        state_layout = ((("x",), 2, 0, torch.float64), (("y",), 1, 2))

        def make_variation_space(self):
            from stable_envs.spaces import Dict

            return Dict({})

        def scene(self, states, variation=None, style=None):  # pragma: no cover - not rendered here
            raise NotImplementedError

    env = F64(num_envs=2, device="cpu")
    rows = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32)
    td = env._unpack_render_state(rows)
    assert td["x"].dtype == torch.float64 and td["y"].dtype == torch.float32
    assert env._pack_render_state(td).dtype == torch.float32 and torch.equal(env._pack_render_state(td), rows)
    assert env.state_spec is not None


def test_unpack_and_state_never_alias_the_env_buffer():
    """A StepResult or ``env.state`` must not change under the caller's feet at the next step."""
    import torch

    from stable_envs.envs.pusht.legacy import PushT

    env = PushT(num_envs=4, device="cpu")
    env.reset(seed=0)
    a = torch.zeros(4, 2)
    out = env.step(a, frames=None)
    snap = out["state"]["agent", "position"].clone()
    st = env.state
    snap2 = st["block", "position"].clone()
    env.step(torch.full((4, 2), 300.0), frames=None)
    assert torch.equal(out["state"]["agent", "position"], snap)
    assert torch.equal(st["block", "position"], snap2)
    td = env._unpack_render_state(env._get_render_state())
    td["agent", "position"].add_(1.0)
    assert not torch.equal(td["agent", "position"], env._unpack_render_state(env._get_render_state())["agent", "position"])


def test_live_state_hook_overrides_the_widened_row():
    import torch

    from stable_envs.env import Env
    from stable_envs.spaces import Dict

    class F64(Env):
        state_dim = 2
        state_names = ("a", "b")
        state_layout = ((("x",), 2, 0, torch.float64),)

        def make_variation_space(self):
            return Dict({})

        def scene(self, states, variation=None, style=None):  # pragma: no cover
            raise NotImplementedError

        def _step(self, action):
            return self._step_result(self.get_state())

        def _live_state(self):
            return {("x",): torch.tensor([[0.1, 0.2]], dtype=torch.float64)}

    env = F64(num_envs=1, device="cpu")
    env._set_render_state(torch.tensor([[0.1, 0.2]], dtype=torch.float32))
    assert torch.equal(env.state["x"], torch.tensor([[0.1, 0.2]], dtype=torch.float64))
    assert not torch.equal(env._unpack_render_state(env._get_render_state())["x"], env.state["x"])
