"""``StyleTensor``, and the one operation three shipped defects went through: ``for_batch``.

"A per-environment appearance silently became environment 0's" has been a shipped defect in this
repository THREE times, in three different files — PushT's ``_var`` doing ``.reshape(-1)[:1].expand(B)``,
the static-prefix bake taking ``scene.kind[:1]``, and ``StableEnv.resolve_style`` falling back to
``self._style.values[:1]`` whenever the render batch differed from the env's. Every one of them passed
the conformance suite of its day, because those checks compare an env against itself under ONE
appearance, where row 3 and row 0 are equal and a collapse is invisible.

``core/style.py`` responded by making the operation singular: there is exactly one function that changes
a style's batch, and a collapse can no longer be reintroduced by editing an env, only by editing that
rule. This file is the test of the rule itself — the level at which the defect actually lives.

It runs on CPU with no env, no renderer and no CUDA, and that is the point: every existing check of this
property (``test_rendering_a_sub_batch_refuses_environment_0_s_style`` and its two neighbours in
``tests/gpu/test_reproducibility.py``) needs a GPU, a built env and a rendered frame to reach a rule that
is twelve lines of tensor arithmetic. Those tests assert that the envs GO THROUGH the rule and must
stay; this asserts what the rule IS.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs import spaces
from stable_envs._style import StyleTensor, rows_are_equal

DEV = "cpu"


def _space():
    """A two-leaf tree: one 3-wide colour and one scalar. Flat width 4, layout sorted by path."""
    return spaces.Dict({
        "block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))}),
        "agent": spaces.Dict({"radius": spaces.Box(low=0.0, high=1.0, shape=(1,),
                                                   init_value=np.array([0.5], np.float32))}),
    })


#: The flat layout is sorted BY PATH, not by declaration order (see ``spaces.Dict.flat_layout``: a
#: layout that shifted with dict insertion order would silently reinterpret every stored style). So for
#: ``_space()`` the row is ``[agent.radius, block.color.r, .g, .b]`` and the fixtures below are written
#: in that order on purpose — writing them in declaration order is the mistake this note prevents.
LAYOUT = [("agent.radius", 0, 1), ("block.color", 1, 3)]


def _style(rows, shared=None):
    v = torch.as_tensor(np.asarray(rows, np.float32), device=DEV).reshape(len(rows), -1)
    assert _space().flat_layout() == LAYOUT, "the flat layout moved; the fixtures below assume it"
    return StyleTensor(v, LAYOUT, shared=shared)


# ------------------------------------------------------------------ rows_are_equal


def test_rows_are_equal_reads_the_VALUES_not_how_the_tensor_was_made():
    """THE bug this function replaced: a strided view whose rows happen to be equal IS batch-uniform.

    ``SceneBuilder`` used to answer "is this batch-uniform?" with ``stride(0) != 0``, which is a test
    for how a tensor was constructed rather than for what it holds. A ``(B, w)`` slice of a real style
    tensor has a non-zero row stride whether or not its rows differ, so the answer was
    "per-environment" for every batched scene ever built and the static-prefix bake it guarded never
    fired above ``B == 1``. Both halves are asserted here: a strided-but-uniform slice is EQUAL, and a
    strided slice that genuinely differs is NOT.
    """
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
    view = base[:, 1:3]
    assert view.stride(0) != 0 and not view.is_contiguous(), (
        "the probe must be a strided, non-contiguous slice or it proves nothing")
    assert rows_are_equal(view)

    differing = base.clone()
    differing[3, 2] = 99.0
    assert not rows_are_equal(differing[:, 1:3])


@pytest.mark.parametrize("t,expected", [
    (None, True),                                          # nothing to disagree
    (torch.tensor(5.0), True),                             # 0-dim: no batch axis
    (torch.zeros(0, 4), True),                             # empty: vacuously uniform
    (torch.zeros(1, 4), True),                             # one row is equal to itself
])
def test_rows_are_equal_degenerate_inputs_are_uniform_not_errors(t, expected):
    """Each of these reaches this function from a real call site and none of them is a disagreement.

    An env with no variation hands it a ``(B, 0)`` tensor; a scalar parameter arrives 0-dim; a
    single-environment render passes one row. Treating any of them as "per-environment" would disable
    the static-prefix bake for exactly the cases it is cheapest on.
    """
    assert rows_are_equal(t) is expected


def test_rows_are_equal_is_the_conjunction_over_everything_it_is_given():
    """It is asked about several tensors at once so the whole question costs ONE device sync."""
    same = torch.ones(4, 3)
    also_same = torch.full((4, 2), 7.0)
    differs = torch.arange(4.0).reshape(4, 1)
    assert rows_are_equal(same, also_same)
    assert not rows_are_equal(same, also_same, differs)
    assert not rows_are_equal(differs, same)


# ------------------------------------------------------------------ for_batch: the rule


def test_a_matching_batch_returns_the_SAME_object_untouched():
    """The render hot path must not pay for this, and must not be able to change the answer.

    Identity rather than equality on purpose: ``for_batch`` is called on every render, and returning a
    copy would both cost an allocation per frame and give the caller a tensor that is no longer the
    env's own, so a later ``set_style`` would silently not be reflected in it.
    """
    s = _style([[1, 2, 3, 0.4], [5, 6, 7, 0.8]])
    assert s.for_batch(2) is s


def test_one_row_broadcasts_to_any_batch():
    """The single-appearance case, which must stay free — it is what ``default_style()`` produces."""
    s = _style([[1, 2, 3, 0.4]], shared=True)
    for b in (1, 3, 4096):
        got = s.for_batch(b)
        assert got.values.shape == (b, 4)
        assert got.shared is True
        assert torch.equal(got.values, s.values.expand(b, 4))
        assert got.values.is_contiguous(), (
            "a broadcast style must be materialised, not left as a stride-0 view: the scene builder "
            "writes into slices of it and a zero-stride destination aliases every row")


def test_B_IDENTICAL_rows_broadcast_because_row_i_IS_row_0():
    """The ergonomic case, and the one a naive "refuse unless one row" rule would break.

    ``set_style(None)`` fills B identical rows, so an env at its default appearance carries a
    ``(B, style_dim)`` tensor rather than a single row. Rendering a sub-batch under it is NOT a
    collapse — row 3 genuinely is row 0 — and it has to keep working without the caller knowing how
    the env stores its default. Sharedness is MEASURED here (``shared=None``), which is the whole
    reason the property holds.
    """
    s = _style([[1, 2, 3, 0.4]] * 8, shared=None)
    got = s.for_batch(1)
    assert got.values.shape == (1, 4)
    assert torch.equal(got.values[0], s.values[0])


def test_B_DIFFERENT_rows_are_REFUSED_rather_than_collapsed_to_environment_0():
    """The defect, asserted as a refusal. Measured before the fix: 229/255, and silent.

        env.set_state(st, style=sty); full = env.render()
        alone = env.render(st[3:4])                            # asked for row 3 only
        |alone - full[3]|.max()                        -> 229  # not row 3's appearance
        |alone - render(st[3:4], style=sty[0:1])|.max() ->   0  # it is row 0's

    There is no correct answer to give here and that is the argument for refusing: a ``(1, state_dim)``
    array says nothing about WHICH row of the env's batch it is, so every choice but an error is a
    guess. The message therefore has to carry three things — both batch sizes, the fact that a shared
    style would have been fine, and the indexed call that IS correct — because a caller who hits this
    is one slice away from the right code and will otherwise reach for ``[:1]``.
    """
    s = _style([[i, i, i, 0.1 * i] for i in range(8)])
    with pytest.raises(ValueError) as e:
        s.for_batch(1)
    msg = str(e.value)
    assert "8" in msg and "1 environment" in msg, f"both batch sizes must be named: {msg}"
    assert "style=env.get_style()[i:j]" in msg, (
        f"the refusal must show the indexed call that works, or the reader will slice with [:1] and "
        f"reintroduce the collapse: {msg}")
    assert "row 0" in msg or "substitute" in msg, (
        f"the message should say what it USED to do, so an old traceback is recognisable: {msg}")


def test_the_owner_of_a_refusal_is_named():
    """``for_batch(owner=)`` puts the caller in the message; a bare ValueError names no file."""
    s = _style([[1, 2, 3, 0.4], [4, 5, 6, 0.7]])
    with pytest.raises(ValueError, match="PushT.render"):
        s.for_batch(1, owner="PushT.render")


# ------------------------------------------------------------------ indexing and conversion


def test_a_leaf_is_addressed_by_path_and_keeps_its_width():
    """``(B, w)`` for a multi-value leaf and ``(B,)`` for a scalar one, because a colour is not a scalar.

    The shape difference is load-bearing rather than cosmetic: env code multiplies a scalar leaf into a
    per-primitive parameter and indexes a vector leaf channel-wise, and a scalar arriving as ``(B, 1)``
    broadcasts into the wrong axis without raising.
    """
    s = _style([[0.4, 1, 2, 3], [0.7, 4, 5, 6]])          # [radius, r, g, b]; see LAYOUT
    assert s["block.color"].shape == (2, 3)
    assert s["agent.radius"].shape == (2,)
    assert torch.equal(s["block.color"][1], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(s["agent.radius"], torch.tensor([0.4, 0.7]))


def test_an_unknown_path_names_the_ones_that_exist():
    s = _style([[1, 2, 3, 0.4]])
    with pytest.raises(KeyError) as e:
        s["block.colour"]                       # British spelling: the plausible mistake
    assert "block.color" in str(e.value), "the error must list the real paths, not just reject one"
    assert s.get("block.colour", "fallback") == "fallback"
    assert "block.color" in s and "nope" not in s


def test_to_preserves_the_layout_and_what_is_already_known_about_sharedness():
    """A device or dtype move must not throw away the answer to a question that cost a sync."""
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
    """``reshape(-1, 0)`` is an error rather than a no-op, so width zero is its own branch.

    Asserted because every path downstream — the scene builder, the datasets, ``for_batch`` — is
    written to take a StyleTensor unconditionally, and an env with no knobs is the case where a
    ``None`` would have been tempting and would have put a branch in all of them.
    """
    s = StyleTensor.from_space(spaces.Dict({}), 7, device=DEV)
    assert s.values.shape == (7, 0) and s.shared is True
    assert s.for_batch(3).values.shape == (3, 0)


@pytest.mark.parametrize("rows", [2, 3, 7])
def test_a_row_count_that_is_neither_one_nor_the_batch_is_refused(rows):
    """The mistake is sliced states with unsliced style, and it used to broadcast instead of raising."""
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
    """Batched domain randomisation, which is the reason any of this is a tensor.

    ``shared`` goes from known-True to measured, and measures False — so the static-prefix bake is
    correctly disabled for this style and ``for_batch`` correctly refuses to reshape it.
    """
    v = np.arange(12, dtype=np.float32).reshape(4, 3)
    s = StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": v})
    assert torch.equal(s["block.color"], torch.as_tensor(v))
    assert s.shared is False
    with pytest.raises(ValueError, match="PER-ENVIRONMENT"):
        s.for_batch(1)


def test_a_per_environment_override_whose_rows_happen_to_agree_is_still_uniform():
    """Measured, not assumed: four identical rows written per-environment still broadcast.

    This is the case that separates "how the value arrived" from "what the value is", and getting it
    wrong in the conservative direction is not harmless — it would refuse a legal sub-batch render for
    a caller who randomised a knob and drew the same number four times.
    """
    v = np.tile(np.array([[1, 2, 3]], np.float32), (4, 1))
    s = StyleTensor.from_space(_space(), 4, device=DEV, overrides={"block.color": v})
    assert s.shared is True
    assert s.for_batch(2).values.shape == (2, 4)


@pytest.mark.parametrize("bad", [2, 5, 11])
def test_an_override_count_that_is_neither_is_refused_by_name(bad):
    """A count that is neither ``w`` nor ``batch*w`` used to be taken as "the first w, broadcast".

    That is environment 0's value for everybody under a third name, and it is the reason the error
    message spells the arithmetic out: the caller who writes ``color=[255, 0]`` or passes a (2, 3)
    array for a batch of 4 is not going to work out ``w`` from a shape mismatch three frames later.
    """
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(_space(), 4, device=DEV,
                               overrides={"block.color": np.zeros(bad, np.float32)})
    msg = str(e.value)
    assert "block.color" in msg and "expected 3" in msg and "12" in msg, msg
    assert "environment 0" in msg, f"the message must name the wrong answer it replaced: {msg}"


def test_an_unknown_override_path_lists_the_real_ones():
    with pytest.raises(KeyError) as e:
        StyleTensor.from_space(_space(), 2, device=DEV, overrides={"blok.color": [1, 2, 3]})
    assert "agent.radius" in str(e.value) and "block.color" in str(e.value)


def test_a_cuda_tensor_of_values_is_accepted_without_a_numpy_round_trip():
    """``np.asarray`` on a CUDA tensor RAISES, so routing values through numpy breaks the common case.

    A caller whose style came out of a dataset batch or a previous render is holding a device tensor.
    Skipped rather than dropped without CUDA: the property is about the device path specifically.
    """
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
