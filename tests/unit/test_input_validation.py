"""Every place user data enters, and the refusal it now gets instead of a plausible picture.

THE DEFECT THIS FILE EXISTS FOR. ``set_state`` did
``torch.as_tensor(states, ...).reshape(-1, self.state_dim)``. Hand that a frame stack and, whenever the
flattened size happens to divide by ``state_dim``, it reshapes without complaint and the env renders a
plausible-looking picture built out of pixel bytes read as coordinates. No exception, no warning, a
frame that looks like a frame. Whether the size divides is arithmetic, not intent — PushT is 5 numbers
against a 96x96x3 frame — so sooner or later it divides.

The rule that replaces it is shape-based rather than size-based, which is what makes it impossible
rather than unlikely: **a state array is 2-D and its last axis is exactly ``state_dim``**. Divisibility
is never consulted for a multi-axis input. ``render`` and ``render_states``, whose documented contract
is "arbitrary leading dims", relax the rank and keep the last-axis rule, which was the load-bearing
half.

THE SAME QUESTION, ASKED EVERYWHERE ELSE. The owner's phrasing is a principle, not a bug report: any
place user data is reshaped, broadcast or index-cast in a way that can silently REINTERPRET it instead
of refusing it. The siblings are covered below, one section each — the style path (which has collapsed
a per-environment quantity to environment 0 three separate times in this repo's history), the two
dataset constructors, ``variation`` dicts with unknown keys, ``sample_style(paths=)``, and ``out=``
buffers.

NO GPU, DELIBERATELY, AND THIS FILE MUST STAY THAT WAY. Shape, dtype, width and finiteness are all
settled before anything touches a renderer, so none of it needs CUDA — and a guard test that lives in
``tests/gpu/`` is SKIPPED on any machine without a device, which reads as coverage while measuring
nothing. This repo has had two incidents of exactly that shape. Everything here runs on CPU tensors
against a ``StableEnv`` subclass built with ``device="cpu"``; the base class's buffers, its style
tensor and its ``__init_subclass__`` render hook are all real, so what is under test is the shipped
path and not a mock of it.

Every check asserts the MESSAGE, not only the exception type. ``pytest.raises(ValueError)`` passes just
as well against ``ValueError("")``, and the entire point of a diagnostic is that a human reads it and
knows what they did and what to do instead.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs import spaces
from stable_envs.env import Env as StableEnv, check_states, require_contiguous_out
from stable_envs._style import StyleTensor
from stable_envs.data import StateDataset, TrajectoryDataset, check_style, render_states

DEV = "cpu"
NAMES = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")


class TinyEnv(StableEnv):
    """5-wide state, a 3x4 frame, one 3-wide appearance knob. Draws WHICH state and WHICH colour.

    A real ``StableEnv`` subclass rather than a duck-typed stand-in, because half of what is under test
    lives in the base class and in the ``__init_subclass__`` hook that wraps every subclass's
    ``render``. Not registered: registration is about the fidelity declarations, and none of this is
    about pixels. Pixel (0,0) carries ``state[0]`` and pixel (0,1) carries the block colour, so "did
    row 3 get row 3's appearance" is an assertion on bytes.
    """

    name = "tiny"
    state_dim = 5
    state_names = NAMES
    default_size = 4

    def make_variation_space(self):
        return spaces.Dict({"block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))})})

    def render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        # The first line of all 26 shipped render overrides, verbatim, including the unchecked
        # ``as_tensor`` + ``reshape`` this file exists to make unreachable. The hook on the base class
        # is what validates and substitutes before this body runs.
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        img = torch.zeros(s.shape[0], 3, 4, 3, dtype=torch.uint8, device=self.device)
        img[:, 0, 0, 0] = s[:, 0].to(torch.uint8)
        img[:, 0, 1, :] = sty["block.color"].to(torch.uint8)
        return img.view(*lead, 3, 4, 3) if lead else img


class ThreeWideEnv(TinyEnv):
    """A 3-wide state, which is what makes a ``(..., 3)`` frame array ambiguous by SHAPE as well.

    The interesting adversary: for this env a frame stack's last axis equals ``state_dim``, so a rule
    that only compared widths would let ``(H, W, 3)`` through as a corpus of states.
    """

    state_dim = 3
    state_names = ("x", "y", "z")


def _env(batch: int = 8, cls=TinyEnv) -> TinyEnv:
    return cls(batch=batch, device=DEV)


def _states(n: int, first: int = 0, width: int = 5) -> np.ndarray:
    """``(n, width)`` whose column 0 is the row's identity, so a frame says which state drew it."""
    out = np.zeros((n, width), np.float32)
    out[:, 0] = np.arange(first, first + n, dtype=np.float32)
    return out


def _ids(frames: torch.Tensor) -> list[int]:
    return frames[:, 0, 0, 0].tolist()


def _colors(frames: torch.Tensor) -> list[int]:
    return frames[:, 0, 1, 0].tolist()


# ============================================================ frames where states were meant


def test_a_frame_stack_whose_size_DIVIDES_is_refused():
    """THE bug. 8*5*5*3 is 600 numbers, an exact multiple of 5, so this used to reshape and RENDER.

    This is the only case in the file where the old code produced pixels rather than a traceback, and
    it is the reason the rule is on the shape rather than on whether the reshape happens to work.
    """
    env = _env(120)                                     # 600 / 5 = 120, so even the batch lined up
    frames = torch.zeros((8, 5, 5, 3), dtype=torch.uint8)
    assert frames.numel() % env.state_dim == 0, "the fixture must be the DIVISIBLE case"
    with pytest.raises(ValueError) as e:
        env.set_state(frames)
    msg = str(e.value)
    assert "FRAMES" in msg, f"the message does not say what arrived: {msg}"
    assert "(8, 5, 5, 3)" in msg, f"the message does not show the input shape: {msg}"
    assert "TinyEnv.set_state" in msg, f"the message does not name the env: {msg}"


def test_a_frame_stack_whose_size_does_NOT_divide_is_refused_with_the_same_message():
    """The loud case, so that the two failures read alike instead of one being a reshape traceback."""
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.set_state(torch.zeros((4, 96, 96, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "FRAMES" in msg and "(4, 96, 96, 3)" in msg, msg
    assert "render()" in msg, f"the message does not point at the direction that works: {msg}"


def test_uint8_is_called_out_by_name_because_states_are_float():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.set_state(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "uint8" in msg and "states are float" in msg, f"dtype is the loudest signal: {msg}"


def test_a_frame_stack_is_refused_even_when_its_last_axis_EQUALS_state_dim():
    """A 3-wide env makes ``(H, W, 3)`` ambiguous by width, so rank has to carry the refusal."""
    env = _env(4, cls=ThreeWideEnv)
    with pytest.raises(ValueError) as e:
        env.set_state(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "FRAMES" in msg or "2-D" in msg, msg


# ============================================================ rank, width, flatness, emptiness


def test_ndim_greater_than_two_is_refused_outright_rather_than_flattened():
    """Even when every axis is innocent: ``(2, 3, 5)`` is a window batch, and it must be said out loud.

    Flattening it silently is the same operation that turned a frame stack into states. The message
    names the explicit reshape, so the legitimate case costs the caller one line and no ambiguity.
    """
    env = _env(6)
    with pytest.raises(ValueError) as e:
        env.set_state(np.stack([_states(3), _states(3, first=10)]))
    msg = str(e.value)
    assert "2-D" in msg and "(2, 3, 5)" in msg, msg
    assert "reshape(-1, 5)" in msg, f"the message does not give the one-line fix: {msg}"


def test_a_row_of_the_wrong_width_names_state_dim_both_numbers_and_the_layout():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.set_state(np.zeros((4, 7), np.float32))
    msg = str(e.value)
    assert "state_dim 5, got 7" in msg, f"the message does not name the field and both widths: {msg}"
    assert "TinyEnv" in msg, f"the message does not name the env: {msg}"
    assert "block_angle" in msg, f"the message does not say what the 5 numbers ARE: {msg}"
    assert "(4, 7)" in msg and "state_names" in msg, msg


def test_a_flat_input_that_is_not_a_whole_number_of_states_is_refused():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.set_state(np.zeros(12, np.float32))
    assert "not a multiple of 5" in str(e.value) and "(12,)" in str(e.value)


def test_a_flat_input_that_is_a_whole_number_of_states_still_works():
    """15 numbers for a 5-wide env is 3 states, and always was. The refusal above is about 12."""
    env = _env(3)
    env.set_state(np.repeat([1.0, 2.0, 3.0], 5).astype(np.float32))
    assert env.get_state().shape == (3, 5)
    assert _ids(env.render()) == [1, 2, 3]


def test_a_scalar_is_refused_with_the_state_layout():
    env = _env(1)
    with pytest.raises(ValueError, match="scalar"):
        env.set_state(3.0)


def test_an_empty_batch_is_refused():
    env = _env(4)
    with pytest.raises(ValueError, match="no states to render"):
        env.set_state(np.zeros((0, 5), np.float32))


# ============================================================ NaN and inf


def test_non_finite_states_name_the_rows_and_the_named_columns():
    """A NaN state renders silently, so the frame looks like data and the loss curve looks like a run.

    The row and the column are both in the message because they answer different questions: a single
    bad row is a bad sample, and a whole bad column is a bad transform.
    """
    env = _env(6)
    st = _states(6)
    st[1, 2] = np.nan
    st[4, 2] = np.inf
    with pytest.raises(ValueError) as e:
        env.set_state(st)
    msg = str(e.value)
    assert "[1, 4]" in msg, f"the message does not say WHICH rows: {msg}"
    assert "block_x" in msg, f"the message does not name the column: {msg}"
    assert "NaN" in msg and "inf" in msg, f"the message does not say which kind: {msg}"
    assert "2 of 6" in msg, f"the message does not say how many of how many: {msg}"


def test_a_single_nan_names_only_its_own_row():
    env = _env(4)
    st = _states(4)
    st[2, 0] = np.nan
    with pytest.raises(ValueError) as e:
        env.set_state(st)
    assert "1 of 4" in str(e.value) and "[2]" in str(e.value) and "agent_x" in str(e.value)


def test_the_non_finite_scan_runs_on_the_caller_s_tensor_before_any_device_copy():
    """Which is what makes it unconditional rather than a flag.

    In the recommended idiom the states arrive from a host-side ``DataLoader``, so the scan is a CPU
    reduction over a few kilobytes and costs no device synchronisation at all. Asserted structurally:
    the reporter is handed a CPU tensor and must answer without moving it.
    """
    from stable_envs.env import _nonfinite_report
    st = torch.zeros(4, 5)
    st[2, 3] = float("nan")
    out = _nonfinite_report(st, NAMES)
    assert out is not None and "[2]" in out and "block_y" in out
    assert _nonfinite_report(torch.zeros(4, 5), NAMES) is None


# ============================================================ things that are not arrays at all


def test_a_trajectory_batch_dict_says_which_key_holds_the_states():
    """``TrajectoryDataset`` yields a dict, and ``torch.as_tensor`` on it dies without naming a key."""
    env = _env(4)
    with pytest.raises(TypeError) as e:
        env.set_state({"states": _states(4), "traj": 0})
    assert "batch['states']" in str(e.value)


def test_the_one_tuple_a_tensordataset_yields_is_accepted_rather_than_refused():
    """``DataLoader(TensorDataset(states))`` yields ``[states]``, which is the recommended idiom's
    most common shape. ``torch.as_tensor`` on it raises "only one element tensors can be converted to
    Python scalars", naming neither the caller nor the fix; stacking it is simply right."""
    env = _env(4)
    env.set_state([torch.as_tensor(_states(4))])
    assert _ids(env.render()) == [0, 1, 2, 3]


def test_something_that_is_not_an_array_at_all_names_the_type_and_what_is_wanted():
    env = _env(2)
    with pytest.raises(TypeError) as e:
        env.set_state(object())
    assert "state_dim" in str(e.value) and "object" in str(e.value)


# ============================================================ the batch rule (short is FINE, long is not)


def test_a_partial_final_batch_is_accepted_and_renders_exactly_those_rows():
    """The normal case under the recommended idiom, not user error.

    Every ``DataLoader`` built with ``drop_last=False`` — torch's default — ends its epoch with a short
    batch, so refusing one makes the ordinary loop the exception. Measured cost of servicing it, on one
    H200: PushT at 96 px renders 4,095 states in 3.371 ms against 3.341 ms for 4,096, and 1,023 in
    2.773 ms against 2.771 for 1,024. Nothing in the renderer or in the static-prefix cache is keyed on
    the batch.
    """
    env = _env(8)
    env.set_state(_states(5))
    assert env.get_state().shape == (5, 5), "get_state must hand back only the rows set_state wrote"
    frames = env.render()
    assert frames.shape[0] == 5, "render() drew the stale tail of the buffer as well as the live rows"
    assert _ids(frames) == [0, 1, 2, 3, 4]


def test_a_full_batch_after_a_short_one_uses_the_whole_buffer_again():
    """``_live_rows`` is per-call state, not a latch. A short batch must not shrink the env."""
    env = _env(8)
    env.set_state(_states(3))
    assert _ids(env.render()) == [0, 1, 2]
    env.set_state(_states(8, first=100))
    assert env.get_state().shape == (8, 5)
    assert _ids(env.render()) == [100, 101, 102, 103, 104, 105, 106, 107]


def test_a_batch_longer_than_the_env_names_both_numbers_and_the_two_fixes():
    """Shorter is fine, longer is not, and the message must not read like the short case's twin."""
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.set_state(_states(9))
    msg = str(e.value)
    assert "4" in msg and "9" in msg, f"the message does not name both batches: {msg}"
    assert "SHORTER" in msg, f"the message does not say which direction IS allowed: {msg}"
    assert "batch=9" in msg, f"the message does not give the constructor fix: {msg}"
    assert "env.render(states)" in msg, f"the message does not give the no-buffer fix: {msg}"


def test_set_state_returns_the_env_so_render_chains_off_it():
    env = _env(4)
    assert env.set_state(_states(4)) is env


# ============================================================ sibling: render(states=)


def test_render_refuses_a_frame_stack_too():
    """``render`` has the same reshape and the same failure; the hook applies the same rule to it.

    It relaxes the RANK, because "arbitrary leading dims" is its documented contract, and keeps the
    last-axis rule, which is the half that was load-bearing.
    """
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env.render(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    assert "FRAMES" in str(e.value) and "TinyEnv.render" in str(e.value)


def test_render_keeps_accepting_leading_dims():
    env = _env(4)
    out = env.render(np.stack([_states(3), _states(3, first=10)]))
    assert tuple(out.shape[:2]) == (2, 3), "render must preserve leading dims, only the width is fixed"


def test_render_refuses_the_wrong_width():
    env = _env(4)
    with pytest.raises(ValueError, match="state_dim 5, got 7"):
        env.render(np.zeros((4, 7), np.float32))


def test_render_takes_any_batch_because_it_does_not_touch_the_env_buffer():
    """The batch rule belongs to ``set_state``, not to the validator, and this pins the difference."""
    env = _env(4)
    assert env.render(_states(9)).shape[0] == 9


# ============================================================ sibling: render_states


def test_render_states_refuses_a_frame_stack():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        render_states(env, torch.zeros((4, 8, 8, 3), dtype=torch.uint8), device=DEV)
    assert "FRAMES" in str(e.value) and "render_states" in str(e.value)


def test_render_states_refuses_a_non_finite_corpus():
    env = _env(4)
    st = _states(4)
    st[3, 1] = np.inf
    with pytest.raises(ValueError) as e:
        render_states(env, st, device=DEV)
    assert "[3]" in str(e.value) and "agent_y" in str(e.value)


# ============================================================ sibling: the style path


def test_a_transposed_style_is_refused_rather_than_regrouped():
    """The style path's version of the same defect, on the quantity that has collapsed three times.

    ``reshape(-1, style_dim)`` succeeds on a ``(style_dim, B)`` array and the row-count check
    downstream then PASSES, because the count is right — every row is a mixture of other environments'
    knobs, and it renders without complaint. Refusing on the SHAPE is what catches it; the exactly
    square case (batch == style_dim) is genuinely ambiguous and no rule can separate it, which is why
    the check is stated as "the last axis is style_dim" rather than as "detect transposes".
    """
    space = TinyEnv(batch=1, device=DEV).variation_space         # style_dim 3
    transposed = np.arange(15, dtype=np.float32).reshape(3, 5)   # meant to be (5, 3)
    assert transposed.size == 5 * 3, "the fixture must hold exactly the right NUMBER of values"
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 5, device=DEV, values=transposed)
    assert "3 number(s) wide" in str(e.value) and "(3, 5)" in str(e.value)


def test_a_style_whose_last_axis_is_the_wrong_width_is_refused():
    space = TinyEnv(batch=1, device=DEV).variation_space
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 4, device=DEV, values=np.zeros((4, 2), np.float32))
    msg = str(e.value)
    assert "3 number(s) wide" in msg and "2" in msg, msg
    assert "style_names" in msg, f"the message does not say where the columns are listed: {msg}"


def test_a_style_with_the_wrong_ROW_count_is_still_refused_by_name():
    space = TinyEnv(batch=1, device=DEV).variation_space
    with pytest.raises(ValueError, match="style has 5 rows but 4 environment"):
        StyleTensor.from_space(space, 4, device=DEV, values=np.zeros((5, 3), np.float32))


def test_a_transposed_per_environment_variation_value_is_refused():
    """``variation={"block.color": (3, B)}`` holds the right NUMBER of values in the wrong arrangement.

    Reshaping it by size gives environment i a mixture of the other environments' colours and renders
    without complaint, which is the same class of defect one level down.
    """
    space = TinyEnv(batch=1, device=DEV).variation_space
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 8, device=DEV,
                               overrides={"block.color": np.zeros((3, 8), np.float32)})
    msg = str(e.value)
    assert "transpose" in msg and "(8, 3)" in msg and "(3, 8)" in msg, msg


def test_an_unknown_variation_key_is_refused_rather_than_ignored():
    space = TinyEnv(batch=1, device=DEV).variation_space
    with pytest.raises(KeyError, match="unknown style path"):
        StyleTensor.from_space(space, 2, device=DEV, overrides={"blcok.color": [1, 2, 3]})


def test_an_unknown_path_in_sample_style_is_refused_rather_than_randomising_nothing():
    """It used to match nothing and hand back the DEFAULT appearance for every environment.

    That is indistinguishable, in the output, from a randomisation study whose randomisation had no
    effect — which is the worst possible way for a typo to present.
    """
    env = _env(4)
    with pytest.raises(KeyError) as e:
        env.sample_style(4, paths=["block.colour"])              # British spelling, silent no-op
    msg = str(e.value)
    assert "block.colour" in msg and "block.color" in msg, f"neither the typo nor the fix: {msg}"
    assert "DEFAULT" in msg, f"the message does not say what used to happen: {msg}"


def test_a_correct_path_in_sample_style_still_randomises():
    env = _env(4)
    got = env.sample_style(4, paths=["block.color"], rng=np.random.default_rng(0))
    assert got.shape == (4, 3) and not np.allclose(got, got[:1])


def test_check_style_accepts_the_two_documented_forms():
    env = _env(4)
    assert check_style(env, np.zeros(3, np.float32), 4, "t").shape == (1, 3)
    assert check_style(env, np.zeros((4, 3), np.float32), 4, "t").shape == (4, 3)


# ============================================================ sibling: the dataset constructors


def test_a_state_dataset_refuses_a_frame_stack():
    """It used to take its width from the INPUT — ``reshape(-1, st.shape[-1])`` — so any array at all
    was a valid corpus and the mismatch surfaced (or did not) much later, at render time."""
    with pytest.raises(ValueError) as e:
        StateDataset(_env(1), torch.zeros((4, 8, 8, 3), dtype=torch.uint8), device=DEV)
    assert "FRAMES" in str(e.value) and "StateDataset" in str(e.value)


def test_a_state_dataset_refuses_the_wrong_state_width():
    with pytest.raises(ValueError, match="state_dim 5, got 2"):
        StateDataset(_env(1), np.zeros((10, 2), np.float32), device=DEV)


def test_a_state_dataset_still_flattens_a_leading_trajectory_axis():
    ds = StateDataset(_env(1), np.stack([_states(4), _states(4, first=10)]), device=DEV)
    assert len(ds) == 8


def test_a_trajectory_dataset_refuses_a_rank_three_frame_for_a_three_wide_env():
    """``(H, W, 3)`` is rank 3, which ``_flatten`` reads as H trajectories of W 3-wide states.

    For an env whose state happens to be 3 numbers wide that used to be accepted in full.
    """
    with pytest.raises(ValueError) as e:
        TrajectoryDataset(_env(1, cls=ThreeWideEnv), torch.zeros((6, 8, 3), dtype=torch.uint8),
                          device=DEV)
    assert "FRAMES" in str(e.value), str(e.value)


def test_a_trajectory_dataset_refuses_a_transposed_style():
    st = np.stack([_states(4), _states(4, first=10)])
    with pytest.raises(ValueError) as e:
        TrajectoryDataset(_env(1), st, style=np.zeros((3, 2), np.float32), device=DEV)
    assert "3 number(s) wide" in str(e.value)


def test_a_trajectory_dataset_still_accepts_a_dense_corpus():
    ds = TrajectoryDataset(_env(1), np.stack([_states(4), _states(4, first=10)]), window=2,
                           device=DEV)
    assert ds.n_trajectories == 2 and len(ds) == 6


# ============================================================ sibling: out= buffers


def test_a_strided_out_buffer_is_refused_by_name():
    """It used to be accepted and SCATTER the frames across the caller's neighbouring slices, because
    ``Tensor.view`` hands a strided tensor straight back when the requested shape is the one it has."""
    buf = torch.zeros((4, 3, 8, 8, 3), dtype=torch.uint8)
    with pytest.raises(ValueError) as e:
        require_contiguous_out("test", buf[:, 1])
    msg = str(e.value)
    assert "contiguous" in msg and "(B, T, H, W, 3)" in msg, msg


def test_a_float_out_buffer_is_refused_for_every_env_not_just_the_rasterizer_ones():
    """The shared guard, so the tile families and the ray-caster get the rasterizer's contract too.

    Those envs reach ``out`` through their own ``view``/``reshape``, which does not object to a float
    destination: the shape is right and the write succeeds, and the caller gets 0..255 floats where
    they asked for pixels. Only the two ``Scene2D`` backends checked it before.
    """
    with pytest.raises(ValueError) as e:
        require_contiguous_out("test", torch.zeros((4, 8, 8, 3), dtype=torch.float32))
    assert "uint8" in str(e.value)
    require_contiguous_out("test", torch.zeros((4, 8, 8, 3), dtype=torch.uint8))     # the good case
    require_contiguous_out("test", None)                                              # and no buffer


def test_an_out_buffer_of_the_wrong_size_or_dtype_is_refused():
    """``_destination`` is the shared contract both rasterizer backends go through."""
    from stable_envs.render.raster2d import _destination
    with pytest.raises(ValueError, match="holds 12 elements"):
        _destination(torch.zeros(12, dtype=torch.uint8), 2, 4, 4)
    with pytest.raises(ValueError, match="must be uint8"):
        _destination(torch.zeros(2 * 4 * 4 * 3, dtype=torch.float32), 2, 4, 4)
    with pytest.raises(TypeError, match="torch tensor"):
        _destination(np.zeros(2 * 4 * 4 * 3, np.uint8), 2, 4, 4)


# ============================================================ the validator itself


def test_check_states_is_reusable_and_does_not_know_about_the_env_batch():
    """The validator is separate from the batch rule on purpose: ``render`` takes any batch.

    So ``check_states`` returns what it was given and lets the caller decide, which is what lets
    ``render(states)`` keep accepting a batch the env's buffer could not hold.
    """
    env = _env(2)
    out = check_states(env, _states(7))
    assert out.shape == (7, 5) and out.dtype == torch.float32


def test_check_states_can_validate_without_moving_a_corpus_to_the_device():
    """``move=False`` is what lets ``render_states`` keep its bounded-peak-memory contract."""
    env = _env(2)
    big = torch.as_tensor(_states(100))
    out = check_states(env, big, "t", leading=True, move=False)
    assert out.device.type == "cpu" and out.dtype == torch.float32
