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
def test_reset_with_rows_matches_reset_and_keeps_other_rows(cls):
    env, ref = cls(), cls()
    options = {"variation": ["block.color"]}
    before = env.reset(seed=1, options=options)
    color = env.get_variation().clone()
    seeds = [None, 40, None, 9]
    state = env.reset(seeds, options, rows=[False, True, False, True])
    _rows_equal(state, before, [0, 2])
    _rows_equal(env.get_variation(), color, [0, 2])
    for i in (1, 3):
        _rows_equal(state, ref.reset(seed=seeds[i] - i, options=options), [i])
        _rows_equal(env.get_variation(), ref.get_variation(), [i])
    _rows_equal(env.get_state(), state, slice(None))


@pytest.mark.parametrize("cls", [Seeded, RowSeeded])
def test_reset_with_rows_int_seed_and_every_row_equal_reset(cls):
    env, ref = cls(), cls()
    _rows_equal(env.reset([7, 8, 9, 10]), ref.reset(seed=7), slice(None))
    env.reset(seed=0)
    _rows_equal(env.reset(7, rows=np.ones(4, bool)), ref.reset(seed=7), slice(None))
    env.reset(seed=0)
    _rows_equal(env.reset(7, rows=[2]), ref.reset(seed=7), [2])
    _rows_equal(env.reset([5, 0, 0, 8], rows=torch.tensor([3, 0])), ref.reset(seed=5), [0, 3])


@pytest.mark.parametrize("cls", [Seeded, RowSeeded])
def test_unseeded_rows_continue_the_stream_before_the_seeded_reset(cls):
    env, ref = cls(), cls()
    env.reset(seed=1)
    ref.reset(seed=1)
    state = env.reset([7, None, 9, None])
    _rows_equal(state, ref.reset(seed=None), [1, 3])
    ref.reset(seed=1)
    _rows_equal(state, ref.reset(seed=[7, 0, 9, 0] if cls.row_seeds else 7), [0, 2])


def test_row_seeds_take_each_row_seed_directly():
    env, ref = RowSeeded(), RowSeeded()
    state = env.reset([0, 0, 0, 0])
    for i in range(4):
        _rows_equal(state[i : i + 1].expand(4), ref.reset(seed=0), [0])


def test_one_batch_seed_needs_seeds_of_at_least_the_row_index():
    env = Seeded()
    env.reset(seed=0)
    held = env.get_state()
    with pytest.raises(ValueError, match=r"rows \[2, 3\] have smaller seeds"):
        env.reset([0, 1, 0, 0])
    _rows_equal(env.get_state(), held, slice(None))


def test_reset_with_rows_checks_its_arguments():
    env = Seeded()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="mask needs shape"):
        env.reset(0, rows=np.ones(3, bool))
    with pytest.raises(ValueError, match="row indices"):
        env.reset(0, rows=[4])
    with pytest.raises(ValueError, match="one entry per row"):
        env.reset([1, 2], rows=[0])
    held = env.get_state()
    _rows_equal(env.reset(3, rows=np.zeros(4, bool)), held, slice(None))


def test_reset_with_rows_keeps_the_default_variation_flag():
    env = Seeded()
    env.reset(seed=0)
    env.reset([0, 5, 0, 0], rows=[1])
    assert env._variation_is_default
    env.reset(seed=0, options={"variation": ["block.color"]})
    env.reset([0, 5, 0, 0], rows=[1])
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
def test_family_reset_with_rows_matches_reset(name, kwargs, options, device):
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
        seeds = [None, 41, None, 7, 1000]
        state = env.reset(seeds, rows=rows, **reset)
        _rows_equal(state, before, ~rows, history)
        _rows_equal(env.get_variation(), variation, ~rows)
        actions = [_actions(env, rng) for _ in range(HORIZON)]
        after = _rollout(env, actions)
        for i in np.flatnonzero(rows.numpy()):
            ref.set_variation(variation)
            ref.set_state(before)
            _rows_equal(state, ref.reset(seed=seeds[i] - i, **reset), [i], history)
            _rows_equal(env.get_variation(), ref.get_variation(), [i])
            for got, expected in zip(after, _rollout(ref, actions), strict=True):
                _rows_equal(got, expected, [i], history)
        ref.set_variation(variation)
        ref.set_state(before)
        for got, expected in zip(after, _rollout(ref, actions), strict=True):
            _rows_equal(got, expected, ~rows, history)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs,options", FAMILIES)
def test_family_per_row_seeds(name, kwargs, options, device):
    with (
        snv.make(name, num_envs=5, device=device, **kwargs) as env,
        snv.make(name, num_envs=5, device=device, **kwargs) as ref,
    ):
        history = env.state_history_leaves
        env.reset(seed=3)
        ref.reset(seed=3)
        seeds = [7, None, 0, None, 2]
        if not env.row_seeds:
            with pytest.raises(ValueError, match="smaller seeds"):
                env.reset(seeds)
            return
        # A row without a seed takes fresh entropy in some families, so only the seeded rows compare.
        state = env.reset(seeds)
        for i in (0, 2, 4):
            ref.reset(seed=3)
            again = ref.reset([s if k == i else None for k, s in enumerate(seeds)], rows=[i])
            _rows_equal(state, again, [i], history)
