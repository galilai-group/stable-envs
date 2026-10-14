"""``StyleTensor`` and ``for_batch``, the single operation that changes a style's batch.

A per-environment appearance must never collapse to environment 0's. Conformance checks under one
appearance cannot see a collapse, so the rule is tested here on CPU with no env or renderer.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs import spaces
from stable_envs._core.style import StyleTensor, rows_are_equal

DEV = "cpu"


def _space():
    """A two-leaf tree: one 3-wide colour and one scalar. Flat width 4, layout sorted by path."""
    return spaces.Dict(
        {
            "block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))}),
            "agent": spaces.Dict(
                {"radius": spaces.Box(low=0.0, high=1.0, shape=(1,), init_value=np.array([0.5], np.float32))}
            ),
        }
    )


#: The flat layout is sorted by path, not declaration order (``spaces.Dict.flat_layout``), so the
#: ``_space()`` row is ``[agent.radius, block.color.r, .g, .b]`` and the fixtures use that order.
LAYOUT = [("agent.radius", 0, 1), ("block.color", 1, 3)]


def _style(rows, shared=None):
    v = torch.as_tensor(np.asarray(rows, np.float32), device=DEV).reshape(len(rows), -1)
    assert _space().flat_layout() == LAYOUT, "the flat layout moved; the fixtures below assume it"
    return StyleTensor(v, LAYOUT, shared=shared)


# ------------------------------------------------------------------ rows_are_equal


def test_rows_are_equal_reads_the_values_not_how_the_tensor_was_made():
    """A strided slice whose rows are equal is batch-uniform; one whose rows differ is not.

    A ``(B, w)`` slice of a real style tensor has a non-zero row stride whether or not its rows
    differ, so a stride test cannot answer the question.
    """
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
    view = base[:, 1:3]
    assert view.stride(0) != 0 and not view.is_contiguous(), (
        "the probe must be a strided, non-contiguous slice or it proves nothing"
    )
    assert rows_are_equal(view)

    differing = base.clone()
    differing[3, 2] = 99.0
    assert not rows_are_equal(differing[:, 1:3])


@pytest.mark.parametrize(
    "t,expected",
    [
        (None, True),  # nothing to disagree
        (torch.tensor(5.0), True),  # 0-dim: no batch axis
        (torch.zeros(0, 4), True),  # empty: vacuously uniform
        (torch.zeros(1, 4), True),  # one row is equal to itself
    ],
)
def test_rows_are_equal_degenerate_inputs_are_uniform_not_errors(t, expected):
    """``None``, 0-dim, ``(B, 0)`` and one-row inputs all reach this from real call sites and are uniform."""
    assert rows_are_equal(t) is expected


def test_rows_are_equal_is_the_conjunction_over_everything_it_is_given():
    """Several tensors are checked together so the question costs one device sync."""
    same = torch.ones(4, 3)
    also_same = torch.full((4, 2), 7.0)
    differs = torch.arange(4.0).reshape(4, 1)
    assert rows_are_equal(same, also_same)
    assert not rows_are_equal(same, also_same, differs)
    assert not rows_are_equal(differs, same)


# ------------------------------------------------------------------ for_batch: the rule


def test_a_matching_batch_returns_the_same_object_untouched():
    """Identity, not equality: a copy would cost an allocation per render and detach from ``set_style``."""
    s = _style([[1, 2, 3, 0.4], [5, 6, 7, 0.8]])
    assert s.for_batch(2) is s


def test_one_row_broadcasts_to_any_batch():
    """The single-appearance case ``default_style()`` produces."""
    s = _style([[1, 2, 3, 0.4]], shared=True)
    for b in (1, 3, 4096):
        got = s.for_batch(b)
        assert got.values.shape == (b, 4)
        assert got.shared is True
        assert torch.equal(got.values, s.values.expand(b, 4))
        assert got.values.is_contiguous(), (
            "a broadcast style must be materialised, not left as a stride-0 view: the scene builder "
            "writes into slices of it and a zero-stride destination aliases every row"
        )


def test_B_IDENTICAL_rows_broadcast_because_row_i_IS_row_0():
    """B identical rows broadcast to a sub-batch; sharedness is measured (``shared=None``).

    ``set_style(None)`` fills B identical rows, so an env at its default appearance carries a
    ``(B, style_dim)`` tensor, and rendering a sub-batch under it is not a collapse.
    """
    s = _style([[1, 2, 3, 0.4]] * 8, shared=None)
    got = s.for_batch(1)
    assert got.values.shape == (1, 4)
    assert torch.equal(got.values[0], s.values[0])


def test_B_DIFFERENT_rows_are_REFUSED_rather_than_collapsed_to_environment_0():
    """Differing rows cannot be narrowed to one environment, so ``for_batch`` refuses.

    A ``(1, state_dim)`` array does not say which row of the batch it is. The message names both batch
    sizes and the indexed call that is correct.
    """
    s = _style([[i, i, i, 0.1 * i] for i in range(8)])
    with pytest.raises(ValueError) as e:
        s.for_batch(1)
    msg = str(e.value)
    assert "8" in msg and "1 environment" in msg, f"both batch sizes must be named: {msg}"
    assert "style=env.get_style()[i:j]" in msg, f"the refusal must show the indexed call that works: {msg}"


def test_the_owner_of_a_refusal_is_named():
    """``for_batch(owner=)`` puts the caller in the message; a bare ValueError names no file."""
    s = _style([[1, 2, 3, 0.4], [4, 5, 6, 0.7]])
    with pytest.raises(ValueError, match="PushT.render"):
        s.for_batch(1, owner="PushT.render")


# ------------------------------------------------------------------ indexing and conversion


def test_a_leaf_is_addressed_by_path_and_keeps_its_width():
    """``(B, w)`` for a multi-value leaf and ``(B,)`` for a scalar one.

    A scalar arriving as ``(B, 1)`` would broadcast into the wrong axis without raising.
    """
    s = _style([[0.4, 1, 2, 3], [0.7, 4, 5, 6]])  # [radius, r, g, b]; see LAYOUT
    assert s["block.color"].shape == (2, 3)
    assert s["agent.radius"].shape == (2,)
    assert torch.equal(s["block.color"][1], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(s["agent.radius"], torch.tensor([0.4, 0.7]))


def test_an_unknown_path_names_the_ones_that_exist():
    s = _style([[1, 2, 3, 0.4]])
    with pytest.raises(KeyError) as e:
        s["block.colour"]  # British spelling: the plausible mistake
    assert "block.color" in str(e.value), "the error must list the real paths, not just reject one"
    assert s.get("block.colour", "fallback") == "fallback"
    assert "block.color" in s and "nope" not in s


def test_to_preserves_the_layout_and_what_is_already_known_about_sharedness():
    """A device or dtype move keeps the known sharedness instead of re-measuring it."""
    s = _style([[1, 2, 3, 0.4]], shared=True)
    m = s.to(dtype=torch.float64)
    assert m.values.dtype == torch.float64
    assert m.paths == s.paths and m.layout == s.layout
    assert m._shared is True, "moving a style re-measured a property it was told at construction"


# ------------------------------------------------------------------ from_space


def test_the_default_is_the_declared_appearance_repeated_per_environment():
    s = StyleTensor.from_space(_space(), 5, device=DEV)
    assert s.values.shape == (5, 4)
    assert s.shared is True
    assert np.allclose(s.values[0].numpy(), _space().pack())


def test_an_env_with_no_variation_still_gets_a_well_formed_style():
    """Width zero is its own branch (``reshape(-1, 0)`` raises) and still yields a StyleTensor."""
    s = StyleTensor.from_space(spaces.Dict({}), 7, device=DEV)
    assert s.values.shape == (7, 0) and s.shared is True
    assert s.for_batch(3).values.shape == (3, 0)


@pytest.mark.parametrize("rows", [2, 3, 7])
def test_a_row_count_that_is_neither_one_nor_the_batch_is_refused(rows):
    """Sliced states with unsliced style must raise, not broadcast."""
    vals = np.zeros((rows, 4), np.float32)
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(_space(), 5, device=DEV, values=vals)
    assert f"{rows} rows" in str(e.value) and "5 environment" in str(e.value)


def test_one_row_of_values_is_repeated_and_known_to_be_shared():
    s = StyleTensor.from_space(_space(), 4, device=DEV, values=np.array([[1, 2, 3, 0.4]], np.float32))
    assert s.values.shape == (4, 4) and s.shared is True


def test_an_override_of_the_leaf_width_is_shared_by_the_batch():
    s = StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": [9, 8, 7]})
    assert torch.equal(s["block.color"], torch.tensor([[9.0, 8.0, 7.0]] * 4))
    assert s.shared is True, "a shared override must not cost the batch its known uniformity"


def test_an_override_of_batch_times_width_is_per_environment():
    """Batched domain randomisation: ``shared`` is measured and is False, so ``for_batch`` refuses."""
    v = np.arange(12, dtype=np.float32).reshape(4, 3)
    s = StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": v})
    assert torch.equal(s["block.color"], torch.as_tensor(v))
    assert s.shared is False
    with pytest.raises(ValueError, match="PER-ENVIRONMENT"):
        s.for_batch(1)


def test_a_per_environment_override_whose_rows_happen_to_agree_is_still_uniform():
    """Four identical rows written per-environment are measured as shared and still broadcast."""
    v = np.tile(np.array([[1, 2, 3]], np.float32), (4, 1))
    s = StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": v})
    assert s.shared is True
    assert s.for_batch(2).values.shape == (2, 4)


@pytest.mark.parametrize("bad", [2, 5, 11])
def test_an_override_count_that_is_neither_is_refused_by_name(bad):
    """A count that is neither ``w`` nor ``batch*w`` is refused with the arithmetic spelled out."""
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": np.zeros(bad, np.float32)})
    msg = str(e.value)
    assert "block.color" in msg and "expected 3" in msg and "12" in msg, msg


def test_an_unknown_override_path_lists_the_real_ones():
    with pytest.raises(KeyError) as e:
        StyleTensor.from_space(_space(), 2, device=DEV, overrides={"blok.color": [1, 2, 3]})
    assert "agent.radius" in str(e.value) and "block.color" in str(e.value)


def test_a_cuda_tensor_of_values_is_accepted_without_a_numpy_round_trip():
    """CUDA values are used directly; ``np.asarray`` on a CUDA tensor raises."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device, so there is no device tensor to hand over")
    v = torch.ones(3, 4, device="cuda")
    s = StyleTensor.from_space(_space(), 3, device="cuda", values=v)
    assert s.values.device.type == "cuda"
    assert torch.equal(s.values, v)


def test_overrides_do_not_write_through_the_caller_s_array():
    """``from_space`` clones before writing; otherwise randomising a knob mutates the caller's buffer."""
    v = np.zeros((4, 3), np.float32)
    StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": v})
    assert not v.any(), "the override array was written through"
