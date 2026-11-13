"""``Spec``/``CompositeSpec``: the four questions a spec answers, and the one it refuses.

CPU only, no device and no reference package. The refusal is the part worth testing: an unbounded
continuous spec does not invent a distribution to sample from, because a state spec with no declared
bounds states that the reachable set is not a box, and the env, not this class, knows it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.specs import CompositeSpec, Spec

pytestmark = pytest.mark.unit


def test_a_spec_describes_shape_dtype_bounds_and_column_names():
    s = Spec(shape=(2,), low=0.0, high=512.0, names=("x", "y"))
    assert s.shape == (2,) and s.numel == 2 and s.bounded
    assert s.batch_shape(7) == (7, 2)
    assert s.zeros(3).shape == (3, 2) and s.zeros(3).dtype is torch.float32
    assert s.names == ("x", "y")


def test_names_must_label_every_element_of_a_one_dimensional_spec():
    with pytest.raises(ValueError, match="names label the elements"):
        Spec(shape=(3,), names=("x", "y"))


def test_sampling_stays_inside_the_bounds_and_is_reproducible_from_a_seed():
    s = Spec(shape=(2,), low=[-1.0, 10.0], high=[1.0, 20.0])
    a = s.sample(64, np.random.default_rng(0))
    b = s.sample(64, np.random.default_rng(0))
    assert torch.equal(a, b)
    assert float(a[:, 0].min()) >= -1.0 and float(a[:, 0].max()) <= 1.0
    assert float(a[:, 1].min()) >= 10.0 and float(a[:, 1].max()) <= 20.0
    assert s.contains(a)


def test_a_discrete_spec_samples_integers_in_range():
    s = Spec(shape=(1,), low=None, high=None, discrete_n=7)
    v = s.sample(256, np.random.default_rng(1))
    assert float(v.min()) >= 0.0 and float(v.max()) <= 6.0
    assert s.contains(v) and not s.contains(torch.full((4, 1), 7.0))


def test_an_unbounded_spec_refuses_to_invent_a_distribution():
    with pytest.raises(ValueError, match="declares no bounds"):
        Spec(shape=(5,)).sample(4)


def test_validate_names_the_columns_when_the_shape_is_wrong():
    s = Spec(shape=(5,), names=tuple("abcde"))
    with pytest.raises(ValueError) as e:
        s.validate(torch.zeros(3, 4), owner="PushT.state")
    assert "PushT.state" in str(e.value) and "'a'" in str(e.value)
    with pytest.raises(ValueError, match="NaN or inf"):
        s.validate(torch.full((2, 5), float("nan")))


def test_the_batch_axis_is_free_and_everything_after_it_is_not():
    s = Spec(shape=(2,), low=-1.0, high=1.0)
    for n in (1, 3, 1024):
        assert s.validate(torch.zeros(n, 2)).shape == (n, 2)


def test_a_composite_mirrors_tensordict_keys_and_samples_a_tensordict():
    spec = CompositeSpec(
        {"agent": {"position": Spec((2,), low=0, high=512)}, "block": {"angle": Spec((1,), low=-3.2, high=3.2)}}
    )
    td = spec.sample(5, np.random.default_rng(0))
    assert list(td.batch_size) == [5]
    assert td["agent", "position"].shape == (5, 2)
    assert spec.keys(nested=True) == [("agent", "position"), ("block", "angle")]
    assert spec["agent", "position"].shape == (2,)
    assert spec.contains(td)
    assert spec.zeros(2)["block", "angle"].abs().sum() == 0


def test_a_composite_refuses_a_state_whose_keys_do_not_match():
    from tensordict import TensorDict

    spec = CompositeSpec({"a": Spec((1,)), "b": Spec((1,))})
    with pytest.raises(ValueError, match="missing"):
        spec.validate(TensorDict({"a": torch.zeros(2, 1)}, batch_size=[2]))
    with pytest.raises(ValueError, match="unexpected"):
        spec.validate(
            TensorDict({"a": torch.zeros(2, 1), "b": torch.zeros(2, 1), "c": torch.zeros(2, 1)}, batch_size=[2])
        )


def test_to_device_is_a_copy_and_leaves_the_original_alone():
    spec = CompositeSpec({"a": Spec((1,))})
    moved = spec.to("cpu")
    assert moved is not spec and moved["a"].device == torch.device("cpu")
