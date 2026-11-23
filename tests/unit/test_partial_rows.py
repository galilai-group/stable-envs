"""``reset(rows=)`` and ``step(rows=)`` change only the selected rows, exactly as a reset or step of every row does."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv
from stable_envs import spaces
from stable_envs._core.seeding import row_seed
from tests.helpers.counter_env import Counter, PipelinedCounter

pytestmark = pytest.mark.unit

#: Steps that the family tests run after a partial reset or step.
HORIZON = 50
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
    ),
]


def _mask(rows, batch):
    if isinstance(rows, slice):
        return torch.ones(batch, dtype=torch.bool)[rows]
    rows = torch.as_tensor(rows)
    if rows.dtype != torch.bool:
        rows = torch.zeros(batch, dtype=torch.bool).index_fill_(0, rows, True)
    return rows


def _rows_equal(actual, expected, rows, history=()):
    """The ``rows`` of two States are bit-identical.

    A ``history`` leaf can be wider in one State than in the other; its extra columns must then be zero padding in
    the selected rows. Every other leaf must have the same shape.
    """
    if torch.is_tensor(actual):
        actual, expected = TensorDict({"x": actual}, actual.shape[:1]), TensorDict({"x": expected}, expected.shape[:1])
    rows = _mask(rows, int(actual.batch_size[0]))
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key in expected.keys(True, True):
        name = "/".join((key,) if isinstance(key, str) else key)
        a, e = actual.get(key)[rows.to(actual.device)], expected.get(key)[rows.to(expected.device)]
        if a.shape != e.shape:
            assert name in history, f"{name}: shape {tuple(a.shape)} != {tuple(e.shape)}"
            width = min(a.shape[1], e.shape[1])
            for wide in (a, e):
                assert not wide[:, width:].any(), f"{name}: the columns past {width} are not padding"
            a, e = a[:, :width], e[:, :width]
        torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True, msg=lambda m, name=name: f"{name}: {m}")


class Seeded(snv.Env):
    """One batch generator seeds every row, as ``reset(seed)`` of a class without ``row_seeds`` does."""

    action_dim = 1
    action_dtype = torch.int64
    reset_sample_paths = frozenset({"block.color"})

    def __init__(self, num_envs=4):
        super().__init__(num_envs=num_envs, device="cpu")
        self.value = torch.zeros(num_envs, 2, dtype=torch.int64)

    def make_variation_space(self):
        return spaces.Dict({"block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))})})

    def _draw(self, seed):
        self._reseed(seed)
        return torch.as_tensor(self.rng.integers(0, 1000, (self.B, 2)))

    def _reset(self, seed=None, options=None):
        self.draw_reset_values(seed, options)
        self.value = self._draw(seed)
        return self.get_state()

    def get_state(self):
        return TensorDict({"value": self.value.clone()}, [self.B])

    def _set_state(self, state):
        self.value = state["value"].clone()

    def _step(self, actions, *, active=None):
        step = torch.as_tensor(actions)
        if active is not None:
            step = torch.where(torch.as_tensor(active)[:, None], step, 0)
        self.value = self.value + step
        return self._step_result(self.get_state(), reward=self.value[:, 0].double())


class RowSeeded(Seeded):
    """Row ``i`` draws from its own seed."""

    row_seeds = True

    def _draw(self, seed):
        if seed is None:
            return torch.as_tensor(self.rng.integers(0, 1000, (self.B, 2)))
        return torch.stack(
            [torch.as_tensor(np.random.default_rng(row_seed(seed, i)).integers(0, 1000, 2)) for i in range(self.B)]
        )


@pytest.mark.parametrize("cls", [Seeded, RowSeeded])
def test_int_seed_with_rows_matches_reset_and_keeps_other_rows(cls):
    env, ref = cls(), cls()
    options = {"variation": ["block.color"]}
    before = env.reset(seed=1, options=options)
    color = env.get_variation().clone()
    state = env.reset(40, options, rows=[False, True, False, True])
    _rows_equal(state, before, [0, 2])
    _rows_equal(env.get_variation(), color, [0, 2])
    _rows_equal(state, ref.reset(seed=40, options=options), [1, 3])
    _rows_equal(env.get_variation(), ref.get_variation(), [1, 3])
    _rows_equal(env.get_state(), state, slice(None))
    _rows_equal(env.reset(7, rows=np.ones(4, bool)), ref.reset(seed=7), slice(None))
    _rows_equal(env.reset(5, rows=torch.tensor([3, 0])), ref.reset(seed=5), [0, 3])


def test_per_row_seeds_need_row_seeds():
    env = Seeded()
    env.reset(seed=0)
    held = env.get_state()
    with pytest.raises(NotImplementedError, match="per-row seeds need row_seeds; pass an int seed"):
        env.reset([0, 1, 2, 3])
    with pytest.raises(NotImplementedError, match="per-row seeds need row_seeds"):
        env.reset([0, 1, 2, 3], rows=[1])
    _rows_equal(env.get_state(), held, slice(None))


def test_row_seeds_match_a_one_row_reset_with_that_seed():
    env = RowSeeded()
    options = {"variation": ["block.color"]}
    env.reset(seed=1, options=options)
    before, color = env.get_state(), env.get_variation().clone()
    seeds = [0, 0, None, 1234]
    state = env.reset(seeds, options, rows=[0, 1, 3])
    for i in (0, 1, 3):
        one = RowSeeded(1)
        _rows_equal(state[i : i + 1], one.reset(seed=seeds[i], options=options), [0])
        _rows_equal(env.get_variation()[i : i + 1], one.get_variation(), [0])
    _rows_equal(state, before, [2])
    _rows_equal(env.get_variation(), color, [2])


def test_unseeded_rows_continue_the_stream_before_the_seeded_reset():
    env, ref = RowSeeded(), RowSeeded()
    env.reset(seed=1)
    ref.reset(seed=1)
    state = env.reset([7, None, 9, None])
    _rows_equal(state, ref.reset(seed=None), [1, 3])


def test_reset_checks_its_arguments():
    env = RowSeeded()
    env.reset(seed=0)
    held = env.get_state()
    with pytest.raises(ValueError, match="mask needs shape"):
        env.reset(0, rows=np.ones(3, bool))
    with pytest.raises(ValueError, match="row indices"):
        env.reset(0, rows=[4])
    with pytest.raises(ValueError, match="one entry per row"):
        env.reset([1, 2], rows=[0])
    for bad in ([0, -1, 0, 0], [0, 1.5, 0, 0], [0, True, 0, 0], [0, 2**63, 0, 0]):
        with pytest.raises(ValueError, match="a row seed must be"):
            env.reset(bad)
    for bad in (1.0, np.array(3), torch.tensor(3), True):
        with pytest.raises(ValueError, match="seed must be None, an int"):
            env.reset(bad)
    _rows_equal(env.get_state(), held, slice(None))
    _rows_equal(env.reset(3, rows=np.zeros(4, bool)), held, slice(None))


def test_a_reset_that_raises_keeps_the_state_and_variation():
    class Failing(RowSeeded):
        fail = False

        def _reset(self, seed=None, options=None):
            out = super()._reset(seed, options)
            if self.fail and seed is not None:
                raise RuntimeError("reset failed")
            return out

    env = Failing()
    env.reset(seed=1, options={"variation": ["block.color"]})
    before, color = env.get_state(), env.get_variation().clone()
    env.fail = True
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset([None, 5, None, 6], rows=[0, 1, 3])
    _rows_equal(env.get_state(), before, slice(None))
    _rows_equal(env.get_variation(), color, slice(None))


def test_reset_with_rows_keeps_the_default_variation_flag():
    env = Seeded()
    env.reset(seed=0)
    env.reset(5, rows=[1])
    assert env._variation_is_default
    env.reset(seed=0, options={"variation": ["block.color"]})
    env.reset(5, rows=[1])
    assert not env._variation_is_default


@pytest.mark.parametrize("cls", [Counter, PipelinedCounter])
@pytest.mark.parametrize("sequence", [False, True])
def test_step_rows_holds_other_rows(cls, sequence):
    env = cls()
    actions = torch.tensor([[1], [2], [3]])
    if sequence:
        actions = actions[:, None].expand(3, 2, 1)
    out = env.step(actions, rows=[False, True, True], frames=slice(None))
    assert out["state"]["position"][:, 0].tolist() == [0, 2 * (1 + sequence), 3 * (1 + sequence)]
    assert env.calls.tolist() == [0, 1 + sequence, 1 + sequence]
    valid = out["valid"]
    assert not valid[0].any() and valid[1:].all()
    assert (out["reward"][0] == 0).all() and not out["frame_valid"][0].any()


@pytest.mark.parametrize("cls", [Counter, PipelinedCounter])
@pytest.mark.parametrize("sequence", [False, True])
def test_step_rows_with_no_row_holds_every_row(cls, sequence):
    env = cls()
    actions = torch.ones(3, *((2,) if sequence else ()), 1, dtype=torch.int64)
    out = env.step(actions, rows=[False] * 3, frames=None)
    assert out["state"]["position"][:, 0].tolist() == [0, 0, 0]
    assert env.calls.tolist() == [0, 0, 0]
    assert not out["valid"].any() and not out["reward"].any()


def test_step_rows_checks_its_arguments():
    env = Counter()
    with pytest.raises(ValueError, match="mask needs shape"):
        env.step(torch.ones(3, 1, dtype=torch.int64), rows=[True] * 2)
    assert env.calls.tolist() == [0, 0, 0]


def _physics_options(env):
    return {"variation": [env.physics_variation_leaves[0]]}


FAMILIES = [
    pytest.param("PushT", {}, {"variation": ["agent.color"]}, marks=pytest.mark.family("pusht"), id="PushT"),
    pytest.param(
        "TwoRoom", {"simulation_backend": "numba"}, None, marks=pytest.mark.family("navigation"), id="TwoRoom"
    ),
    pytest.param("CartPole", {}, None, marks=pytest.mark.family("classic"), id="CartPole"),
    pytest.param("DMCReacher", {}, _physics_options, marks=pytest.mark.family("dmc"), id="DMCReacher"),
    pytest.param("FetchReach", {}, None, marks=pytest.mark.family("fetch"), id="FetchReach"),
    pytest.param("CraftaxClassicSymbolic", {}, None, marks=pytest.mark.family("craftax"), id="Craftax"),
    pytest.param(
        "PyFlytRocketLanding",
        {"simulation_backend": "numba"},
        None,
        marks=pytest.mark.family("rocket"),
        id="PyFlytRocketLanding",
    ),
]
STEP_FAMILIES = [
    *FAMILIES,
    pytest.param("MSHumanLocomotion", {}, None, marks=pytest.mark.family("mshuman"), id="MSHumanLocomotion"),
]


def _actions(env, rng, length=None):
    batch = env.action_spec.sample(env.num_envs, rng=rng)
    if length is None:
        return batch
    return torch.stack([env.action_spec.sample(env.num_envs, rng=rng) for _ in range(length)], dim=1)


def _reset_kwargs(env, options):
    if options is None:
        return {}
    return {"options": options(env) if callable(options) else options}


def _rollout(env, actions):
    return [env.step(action, states=None, frames=None)["state"] for action in actions]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs,options", STEP_FAMILIES)
@pytest.mark.parametrize("length", [None, 3])
def test_family_step_rows_matches_full_step(name, kwargs, options, length, device):
    rng = np.random.default_rng(0)
    with (
        snv.make(name, num_envs=5, device=device, **kwargs) as env,
        snv.make(name, num_envs=5, device=device, **kwargs) as ref,
    ):
        history = env.state_history_leaves
        env.reset(seed=3)
        for _ in range(2):
            env.step(_actions(env, rng), frames=None)
        rows = torch.tensor([True, False, True, True, False])
        for _ in range(HORIZON if length is None else 4):
            before = env.get_state()
            ref.set_state(before)
            actions = _actions(env, rng, length)
            out = env.step(actions, rows=rows, frames=None)
            full = ref.step(actions, frames=None)
            _rows_equal(out["state"], full["state"], rows, history)
            _rows_equal(out["state"], before, ~rows, history)
            _rows_equal(env.get_state(), out["state"], slice(None))
            assert not out["valid"][~rows].any()
            for key in ("reward", "terminated"):
                if key in full.keys():
                    _rows_equal(out[key], full[key], rows)
                    assert not out[key][~rows].any()
        before = env.get_state()
        out = env.step(_actions(env, rng), rows=[], frames=None)
        _rows_equal(out["state"], before, slice(None), history)
        assert not out["valid"].any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs,options", FAMILIES)
def test_family_int_seed_with_rows_matches_reset(name, kwargs, options, device):
    rng = np.random.default_rng(1)
    with (
        snv.make(name, num_envs=5, device=device, **kwargs) as env,
        snv.make(name, num_envs=5, device=device, **kwargs) as ref,
    ):
        history = env.state_history_leaves
        reset = _reset_kwargs(env, options)
        env.reset(seed=3, **reset)
        for _ in range(3):
            env.step(_actions(env, rng), frames=None)
        before, variation = env.get_state(), env.get_variation().clone()
        rows = torch.tensor([False, True, False, True, True])
        state = env.reset(41, rows=rows, **reset)
        _rows_equal(state, before, ~rows, history)
        _rows_equal(env.get_variation(), variation, ~rows)
        ref.set_variation(variation)
        ref.set_state(before)
        _rows_equal(state, ref.reset(seed=41, **reset), rows, history)
        _rows_equal(env.get_variation(), ref.get_variation(), rows)
        actions = [_actions(env, rng) for _ in range(HORIZON)]
        after = _rollout(env, actions)
        for got, expected in zip(after, _rollout(ref, actions), strict=True):
            _rows_equal(got, expected, rows, history)
        ref.set_variation(variation)
        ref.set_state(before)
        for got, expected in zip(after, _rollout(ref, actions), strict=True):
            _rows_equal(got, expected, ~rows, history)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs,options", FAMILIES)
def test_family_per_row_seeds_match_a_one_row_reset(name, kwargs, options, device):
    rng = np.random.default_rng(4)
    with snv.make(name, num_envs=5, device=device, **kwargs) as env:
        history = env.state_history_leaves
        reset = _reset_kwargs(env, options)
        env.reset(seed=3, **reset)
        seeds = [7, None, 0, 7, 2**20]
        if not env.row_seeds:
            with pytest.raises(NotImplementedError, match="per-row seeds need row_seeds"):
                env.reset(seeds, **reset)
            return
        before = env.get_state()
        state = env.reset(seeds, rows=[0, 2, 3, 4], **reset)
        _rows_equal(state, before, [1], history)
        actions = [_actions(env, rng) for _ in range(20)]
        after = _rollout(env, actions)
        for i in (0, 2, 3, 4):
            with snv.make(name, num_envs=1, device=device, **kwargs) as one:
                _rows_equal(state[i : i + 1], one.reset(seed=seeds[i], **reset), [0], history)
                _rows_equal(env.get_variation()[i : i + 1], one.get_variation(), [0])
                for got, action in zip(after, actions, strict=True):
                    expected = one.step(action[i : i + 1], states=None, frames=None)["state"]
                    _rows_equal(got[i : i + 1], expected, [0], history)


@pytest.mark.parametrize("name,kwargs,options", [p for p in FAMILIES if p.id == "PushT"])
def test_a_row_without_a_seed_resets_with_its_own_stream(name, kwargs, options):
    with snv.make(name, num_envs=5, device="cpu", **kwargs) as env:
        history = env.state_history_leaves
        env.reset(seed=3)
        before = env.get_state()
        state = env.reset([None, 5, None, None, None], rows=[0, 1])
        _rows_equal(state, before, [2, 3, 4], history)
        with snv.make(name, num_envs=1, device="cpu", **kwargs) as one:
            _rows_equal(state[1:2], one.reset(seed=5), [0], history)


@pytest.mark.family("navigation")
def test_frames_after_partial_reset_and_step():
    rng = np.random.default_rng(5)
    with (
        snv.make("TwoRoom", num_envs=4, device="cpu", simulation_backend="numba") as env,
        snv.make("TwoRoom", num_envs=4, device="cpu", simulation_backend="numba") as ref,
    ):
        env.reset(seed=3)
        before = env.get_state()
        rows = torch.tensor([True, False, True, False])
        env.reset(9, rows=rows)
        ref.set_state(before)
        ref.reset(seed=9)
        _rows_equal(env.render(), ref.render(), rows)
        _rows_equal(env.render(), ref.render(before), ~rows)
        before = env.get_state()
        ref.set_state(before)
        actions = _actions(env, rng)
        out = env.step(actions, rows=rows, frames=slice(-1, None))
        full = ref.step(actions, frames=slice(-1, None))
        _rows_equal(out["frames"][:, -1], full["frames"][:, -1], rows)
        _rows_equal(out["frames"][:, -1], env.render(before), ~rows)
