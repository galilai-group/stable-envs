"""``reset(rows=)`` and ``step(rows=)`` change only the selected rows, exactly as a reset or step of every row does."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv
from stable_envs import spaces
from tests.helpers.counter_env import Counter, PipelinedCounter

pytestmark = pytest.mark.unit


def _rows_equal(actual, expected, rows):
    """The ``rows`` of two States are bit-identical; a leaf whose second axis differs compares its common part."""
    if isinstance(rows, slice):
        rows = torch.ones(actual.shape[0], dtype=torch.bool)[rows]
    rows = torch.as_tensor(rows)
    if torch.is_tensor(actual):
        actual, expected = TensorDict({"x": actual}, actual.shape[:1]), TensorDict({"x": expected}, expected.shape[:1])
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key in expected.keys(True, True):
        a, e = actual.get(key)[rows.to(actual.device)], expected.get(key)[rows.to(expected.device)]
        if a.shape != e.shape and a.ndim >= 2:
            width = min(a.shape[1], e.shape[1])
            a, e = a[:, :width], e[:, :width]
        torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True, msg=lambda m, key=key: f"{key}: {m}")


class Seeded(snv.Env):
    """Each row's value depends on the reset seed and the row index; one color leaf is sampled per row."""

    action_dim = 1
    action_dtype = torch.int64
    reset_sample_paths = frozenset({"block.color"})

    def __init__(self, num_envs=4):
        super().__init__(num_envs=num_envs, device="cpu")
        self.value = torch.zeros(num_envs, 2, dtype=torch.int64)

    def make_variation_space(self):
        return spaces.Dict({"block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))})})

    def _reset(self, seed=None, options=None):
        self.draw_reset_values(seed, options)
        rng = np.random.default_rng(seed)
        self.value = torch.as_tensor(rng.integers(0, 1000, (self.B, 2)))
        return self.get_state()

    def get_state(self):
        return TensorDict({"value": self.value.clone()}, [self.B])

    def _set_state(self, state):
        self.value = state["value"].clone()

    def _step(self, actions):
        self.value = self.value + torch.as_tensor(actions)
        return self._step_result(self.get_state(), reward=self.value[:, 0].double())


def test_reset_with_rows_matches_reset_and_keeps_other_rows():
    env, ref = Seeded(), Seeded()
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


def test_reset_with_rows_int_seed_and_every_row_equal_reset():
    env, ref = Seeded(), Seeded()
    _rows_equal(env.reset([7, 8, 9, 10]), ref.reset(seed=7), slice(None))
    env.reset(seed=0)
    _rows_equal(env.reset(7, rows=np.ones(4, bool)), ref.reset(seed=7), slice(None))
    env.reset(seed=0)
    _rows_equal(env.reset(7, rows=[2]), ref.reset(seed=7), [2])
    _rows_equal(env.reset([5, 0, 0, 8], rows=torch.tensor([3, 0])), ref.reset(seed=5), [0, 3])


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


def test_step_rows_checks_its_arguments():
    env = Counter()
    with pytest.raises(ValueError, match="selects no row"):
        env.step(torch.ones(3, 1, dtype=torch.int64), rows=[False] * 3)
    with pytest.raises(ValueError, match="mask needs shape"):
        env.step(torch.ones(3, 1, dtype=torch.int64), rows=[True] * 2)
    assert env.calls.tolist() == [0, 0, 0]


def test_step_rows_without_an_active_mask_restores_other_rows():
    env = Seeded(3)
    env.reset(seed=2)
    before = env.get_state()
    out = env.step(torch.tensor([[1], [2], [3]]), rows=[1], frames=None)
    assert (out["state"]["value"] - before["value"])[:, 0].tolist() == [0, 2, 0]
    assert out["valid"][:, 0].tolist() == [False, True, False]
    assert out["reward"][1] == before["value"][1, 0] + 2 and out["reward"][0] == 0


FAMILIES = [
    pytest.param("PushT", {}, {"variation": ["agent.color"]}, marks=pytest.mark.family("pusht"), id="PushT"),
    pytest.param(
        "TwoRoom", {"simulation_backend": "numba"}, None, marks=pytest.mark.family("navigation"), id="TwoRoom"
    ),
    pytest.param("CartPole", {}, None, marks=pytest.mark.family("classic"), id="CartPole"),
]


def _actions(env, rng, length=None):
    batch = env.action_spec.sample(env.num_envs, rng=rng)
    if length is None:
        return batch
    return torch.stack([env.action_spec.sample(env.num_envs, rng=rng) for _ in range(length)], dim=1)


@pytest.mark.parametrize("name,kwargs,options", FAMILIES)
@pytest.mark.parametrize("length", [None, 3])
def test_family_step_rows_matches_full_step(name, kwargs, options, length):
    rng = np.random.default_rng(0)
    with (
        snv.make(name, num_envs=5, device="cpu", **kwargs) as env,
        snv.make(name, num_envs=5, device="cpu", **kwargs) as ref,
    ):
        env.reset(seed=3)
        for _ in range(2):
            env.step(_actions(env, rng), frames=None)
        rows = torch.tensor([True, False, True, True, False])
        for _ in range(3):
            before = env.get_state()
            ref.set_state(before)
            actions = _actions(env, rng, length)
            out = env.step(actions, rows=rows, frames=None)
            full = ref.step(actions, frames=None)
            _rows_equal(out["state"], full["state"], rows)
            _rows_equal(out["state"], before, ~rows)
            _rows_equal(env.get_state(), out["state"], slice(None))
            assert not out["valid"][~rows].any()
            for key in ("reward", "terminated"):
                _rows_equal(out[key], full[key], rows)
                assert not out[key][~rows].any()


@pytest.mark.parametrize("name,kwargs,options", FAMILIES)
def test_family_reset_with_rows_matches_reset(name, kwargs, options):
    rng = np.random.default_rng(1)
    reset = {} if options is None else {"options": options}
    with (
        snv.make(name, num_envs=5, device="cpu", **kwargs) as env,
        snv.make(name, num_envs=5, device="cpu", **kwargs) as ref,
    ):
        env.reset(seed=3, **reset)
        for _ in range(3):
            env.step(_actions(env, rng), frames=None)
        before, variation = env.get_state(), env.get_variation().clone()
        rows = torch.tensor([False, True, False, True, True])
        seeds = [None, 41, None, 7, 1000]
        state = env.reset(seeds, rows=rows, **reset)
        _rows_equal(state, before, ~rows)
        _rows_equal(env.get_variation(), variation, ~rows)
        actions = _actions(env, rng)
        after = env.step(actions, frames=None)["state"]
        for i in np.flatnonzero(rows.numpy()):
            ref.set_variation(variation)
            ref.set_state(before)
            _rows_equal(state, ref.reset(seed=seeds[i] - i, **reset), [i])
            _rows_equal(env.get_variation(), ref.get_variation(), [i])
            _rows_equal(after, ref.step(actions, frames=None)["state"], [i])
        ref.set_variation(variation)
        ref.set_state(before)
        _rows_equal(after, ref.step(actions, frames=None)["state"], ~rows)
