"""Variation Dict sampling order, selective update and exact leaves, against SWM's own spaces."""

from __future__ import annotations

import numpy as np
import pytest

from stable_envs import spaces as ours

pytestmark = pytest.mark.unit


def _tree(mod, *, order):
    return mod.Dict(
        {
            "a": mod.Dict(
                {
                    "x": mod.Box(0.0, 1.0, shape=(2,), init_value=np.zeros(2, np.float32)),
                    "k": mod.Discrete(5, init_value=0),
                }
            ),
            "b": mod.Dict({"c": mod.RGBBox(init_value=np.zeros(3, np.uint8))}),
            "z": mod.Box(-1.0, 1.0, shape=(), init_value=np.float32(0)),
        },
        sampling_order=order,
    )


def _leaf(space, dotted):
    for part in dotted.split("."):
        space = space[part]
    return space


def _values(space):
    return {name: np.asarray(_leaf(space, name).value).tolist() for name in sorted(space.names())}


@pytest.fixture(scope="module")
def swm_spaces():
    return pytest.importorskip("stable_worldmodel.spaces")


@pytest.mark.swm
@pytest.mark.family("classic")
@pytest.mark.parametrize("keys", [["all"], ["z", "a.x", "b.c"], ["a"], ["a.k", "z"]])
def test_update_draws_match_swm_when_a_sampling_order_is_given(swm_spaces, keys):
    order = ["z", "b", "a"]
    mine, theirs = _tree(ours, order=order), _tree(swm_spaces, order=order)
    mine.seed(11)
    theirs.seed(11)
    mine.update(keys)
    theirs.update(keys)
    assert _values(mine) == _values(theirs)
    assert any(v != _values(_tree(ours, order=order))[k] for k, v in _values(mine).items())


@pytest.mark.swm
@pytest.mark.family("classic")
def test_sample_returns_the_dict_in_sampling_order_and_stores_it(swm_spaces):
    order = ["z", "b", "a"]
    mine, theirs = _tree(ours, order=order), _tree(swm_spaces, order=order)
    mine.seed(3)
    theirs.seed(3)
    got, want = mine.sample(), theirs.sample()
    assert list(got) == order
    assert np.array_equal(got["z"], want["z"])
    assert np.array_equal(got["a"]["x"], want["a"]["x"])
    assert _values(mine) == _values(theirs)


def test_leaves_are_drawn_in_the_requested_order():
    drawn = []

    class Recording(ours.Box):
        def __init__(self, name):
            super().__init__(0.0, 1.0, shape=())
            self.name = name

        def sample(self, *args, **kw):
            drawn.append(self.name)
            return super().sample(*args, **kw)

    space = ours.Dict({n: Recording(n) for n in "abc"}, sampling_order=["c", "a", "b"])
    space.update(["all"])
    assert drawn == ["c", "a", "b"]
    drawn.clear()
    space.update(["b", "c", "a"])
    assert drawn == ["c", "a", "b"]


def test_unlisted_keys_sample_after_the_listed_ones():
    space = _tree(ours, order=["z"])
    assert list(space.sample()) == ["z", "a", "b"]


def test_unknown_sampling_order_key_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        _tree(ours, order=["nope"])


def test_exact_box_keeps_the_python_float_but_the_default_casts_to_dtype():
    exact = ours.Box(0.7, 0.95, shape=(), init_value=0.9, exact=True)
    cast = ours.Box(0.7, 0.95, shape=(), init_value=0.9)
    assert isinstance(exact.value, float) and exact.value == 0.9
    assert float(cast.value) != 0.9
    assert cast.value == np.float32(0.9)


def test_set_init_value_can_leave_the_current_value_alone():
    box = ours.Box(0.0, 10.0, shape=(), init_value=1.0)
    box.set_init_value(4.0, reset_value=False)
    assert float(box.value) == 1.0
    assert float(box.init_value) == 4.0
    box.reset()
    assert float(box.value) == 4.0
