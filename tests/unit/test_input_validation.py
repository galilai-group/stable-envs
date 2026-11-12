"""User data entering the env is refused when it could be silently reinterpreted.

``set_state`` must not reshape a frame stack into states just because its size divides ``state_dim``
(PushT is 5 numbers against a 96x96x3 frame). The rule is shape-based: a state array is 2-D and its
last axis is exactly ``state_dim``; ``render`` and ``render_states`` relax the rank and keep the
last-axis rule. The same principle is checked for the style path, the dataset constructors, unknown
``variation`` keys, ``sample_style(paths=)`` and ``out=`` buffers.

Everything runs on CPU against a real ``Env`` subclass, so the base-class buffers, style tensor and
render hook are the shipped ones. Each check asserts the message, not only the exception type.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs._core.style import StyleTensor
from stable_envs.env import check_states, require_contiguous_out
from stable_envs.testing._data import StateDataset, TrajectoryDataset, check_style, render_states
from tests.helpers.tiny_env import NAMES, TinyEnv, frame_ids, id_states

DEV = "cpu"


class ThreeWideEnv(TinyEnv):
    """A 3-wide state: a frame stack's last axis equals ``state_dim``, so a width-only rule would admit it."""

    state_dim = 3
    state_names = ("x", "y", "z")


def _env(num_envs: int = 8, cls=TinyEnv) -> TinyEnv:
    return cls(num_envs=num_envs, device=DEV)


def _colors(frames: torch.Tensor) -> list[int]:
    return frames[:, 0, 1, 0].tolist()


# ============================================================ frames where states were meant


def test_a_frame_stack_whose_size_divides_is_refused():
    """8*5*5*3 is 600 numbers, a multiple of 5, so a size-based rule would reshape and render it."""
    env = _env(120)  # 600 / 5 = 120, so even the batch lined up
    frames = torch.zeros((8, 5, 5, 3), dtype=torch.uint8)
    assert frames.numel() % env.state_dim == 0, "the fixture size must divide by state_dim"
    with pytest.raises(ValueError) as e:
        env._set_render_state(frames)
    msg = str(e.value)
    assert "FRAMES" in msg, f"the message does not say what arrived: {msg}"
    assert "(8, 5, 5, 3)" in msg, f"the message does not show the input shape: {msg}"
    assert "TinyEnv.set_state" in msg, f"the message does not name the env: {msg}"


def test_a_frame_stack_whose_size_does_not_divide_is_refused_with_the_same_message():
    """A non-dividing frame stack fails with the same message as a dividing one."""
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._set_render_state(torch.zeros((4, 96, 96, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "FRAMES" in msg and "(4, 96, 96, 3)" in msg, msg
    assert "not image pixels" in msg, f"the message does not say what to supply instead: {msg}"


def test_uint8_is_called_out_by_name_because_states_are_float():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._set_render_state(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "uint8" in msg and "astype(" in msg, f"the message does not name uint8 or suggest astype: {msg}"


def test_a_frame_stack_is_refused_even_when_its_last_axis_equals_state_dim():
    """A 3-wide env makes ``(H, W, 3)`` ambiguous by width, so rank has to carry the refusal."""
    env = _env(4, cls=ThreeWideEnv)
    with pytest.raises(ValueError) as e:
        env._set_render_state(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    msg = str(e.value)
    assert "FRAMES" in msg or "2-D" in msg, msg


# ============================================================ rank, width, flatness, emptiness


def test_ndim_greater_than_two_is_refused_outright_rather_than_flattened():
    """A ``(2, 3, 5)`` window batch is refused, and the message names the explicit reshape."""
    env = _env(6)
    with pytest.raises(ValueError) as e:
        env._set_render_state(np.stack([id_states(3), id_states(3, first=10)]))
    msg = str(e.value)
    assert "2-D" in msg and "(2, 3, 5)" in msg, msg
    assert "reshape to (-1, 5)" in msg, f"the message does not give the one-line fix: {msg}"


def test_a_row_of_the_wrong_width_names_state_dim_both_numbers_and_the_layout():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._set_render_state(np.zeros((4, 7), np.float32))
    msg = str(e.value)
    assert "state_dim 5, got 7" in msg, f"the message does not name the field and both widths: {msg}"
    assert "TinyEnv" in msg, f"the message does not name the env: {msg}"
    assert "block_angle" in msg, f"the message does not list the state names: {msg}"
    assert "(4, 7)" in msg and "state_names" in msg, msg


def test_a_flat_input_that_is_not_a_whole_number_of_states_is_refused():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._set_render_state(np.zeros(12, np.float32))
    assert "not a multiple of 5" in str(e.value) and "(12,)" in str(e.value)


def test_a_flat_input_that_is_a_whole_number_of_states_still_works():
    """15 numbers for a 5-wide env are 3 states."""
    env = _env(3)
    env._set_render_state(np.repeat([1.0, 2.0, 3.0], 5).astype(np.float32))
    assert env._get_render_state().shape == (3, 5)
    assert frame_ids(env._render()) == [1, 2, 3]


def test_a_scalar_is_refused_with_the_state_layout():
    env = _env(1)
    with pytest.raises(ValueError, match="scalar"):
        env._set_render_state(3.0)


def test_an_empty_batch_is_refused():
    env = _env(4)
    with pytest.raises(ValueError, match="no states to render"):
        env._set_render_state(np.zeros((0, 5), np.float32))


# ============================================================ NaN and inf


def test_non_finite_states_name_the_rows_and_the_named_columns():
    """Non-finite states are refused; the message names the rows, the columns and the kind."""
    env = _env(6)
    st = id_states(6)
    st[1, 2] = np.nan
    st[4, 2] = np.inf
    with pytest.raises(ValueError) as e:
        env._set_render_state(st)
    msg = str(e.value)
    assert "[1, 4]" in msg, f"the message does not say which rows: {msg}"
    assert "block_x" in msg, f"the message does not name the column: {msg}"
    assert "NaN" in msg and "inf" in msg, f"the message does not say which kind: {msg}"
    assert "2 of 6" in msg, f"the message does not say how many of how many: {msg}"


def test_a_single_nan_names_only_its_own_row():
    env = _env(4)
    st = id_states(4)
    st[2, 0] = np.nan
    with pytest.raises(ValueError) as e:
        env._set_render_state(st)
    assert "1 of 4" in str(e.value) and "[2]" in str(e.value) and "agent_x" in str(e.value)


def test_the_non_finite_scan_runs_on_the_caller_s_tensor_before_any_device_copy():
    """The non-finite reporter answers on a CPU tensor without moving it to a device."""
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
        env._set_render_state({"states": id_states(4), "traj": 0})
    assert "batch['states']" in str(e.value)


def test_the_one_tuple_a_tensordataset_yields_is_accepted_rather_than_refused():
    """The one-element list ``DataLoader(TensorDataset(states))`` yields is accepted."""
    env = _env(4)
    env._set_render_state([torch.as_tensor(id_states(4))])
    assert frame_ids(env._render()) == [0, 1, 2, 3]


def test_something_that_is_not_an_array_at_all_names_the_type_and_what_is_wanted():
    env = _env(2)
    with pytest.raises(TypeError) as e:
        env._set_render_state(object())
    assert "state_dim" in str(e.value) and "object" in str(e.value)


# ============================================================ the batch rule (short is FINE, long is not)


def test_a_partial_final_batch_is_accepted_and_renders_exactly_those_rows():
    """A short final batch (``drop_last=False``, torch's default) is accepted and renders only its rows."""
    env = _env(8)
    env._set_render_state(id_states(5))
    assert env._get_render_state().shape == (5, 5), "get_state must hand back only the rows set_state wrote"
    frames = env._render()
    assert frames.shape[0] == 5, "render() drew the stale tail of the buffer as well as the live rows"
    assert frame_ids(frames) == [0, 1, 2, 3, 4]


def test_a_full_batch_after_a_short_one_uses_the_whole_buffer_again():
    """``_live_rows`` is per-call state, not a latch. A short batch must not shrink the env."""
    env = _env(8)
    env._set_render_state(id_states(3))
    assert frame_ids(env._render()) == [0, 1, 2]
    env._set_render_state(id_states(8, first=100))
    assert env._get_render_state().shape == (8, 5)
    assert frame_ids(env._render()) == [100, 101, 102, 103, 104, 105, 106, 107]


def test_a_batch_longer_than_the_env_names_both_numbers_and_the_two_fixes():
    """Shorter is fine, longer is not, and the message must not read like the short case's twin."""
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._set_render_state(id_states(9))
    msg = str(e.value)
    assert "4" in msg and "9" in msg, f"the message does not name both batches: {msg}"
    assert "SHORTER" in msg, f"the message does not say which direction is allowed: {msg}"
    assert "num_envs=9" in msg, f"the message does not give the constructor fix: {msg}"
    assert "env._render(states)" in msg, f"the message does not give the no-buffer fix: {msg}"


def test_set_state_returns_the_env_so_render_chains_off_it():
    env = _env(4)
    assert env._set_render_state(id_states(4)) is env


# ============================================================ sibling: render(states=)


def test_render_refuses_a_frame_stack_too():
    """``render`` refuses a frame stack; it accepts any leading dims but keeps the last-axis rule."""
    env = _env(4)
    with pytest.raises(ValueError) as e:
        env._render(torch.zeros((4, 8, 8, 3), dtype=torch.uint8))
    assert "FRAMES" in str(e.value) and "TinyEnv.render" in str(e.value)


def test_render_keeps_accepting_leading_dims():
    env = _env(4)
    out = env._render(np.stack([id_states(3), id_states(3, first=10)]))
    assert tuple(out.shape[:2]) == (2, 3), "render must preserve leading dims, only the width is fixed"


def test_render_refuses_the_wrong_width():
    env = _env(4)
    with pytest.raises(ValueError, match="state_dim 5, got 7"):
        env._render(np.zeros((4, 7), np.float32))


def test_render_takes_any_batch_because_it_does_not_touch_the_env_buffer():
    """The batch rule belongs to ``set_state``, not to the validator, and this pins the difference."""
    env = _env(4)
    assert env._render(id_states(9)).shape[0] == 9


# ============================================================ sibling: render_states


def test_render_states_refuses_a_frame_stack():
    env = _env(4)
    with pytest.raises(ValueError) as e:
        render_states(env, torch.zeros((4, 8, 8, 3), dtype=torch.uint8), device=DEV)
    assert "FRAMES" in str(e.value) and "render_states" in str(e.value)


def test_render_states_refuses_a_non_finite_corpus():
    env = _env(4)
    st = id_states(4)
    st[3, 1] = np.inf
    with pytest.raises(ValueError) as e:
        render_states(env, st, device=DEV)
    assert "[3]" in str(e.value) and "agent_y" in str(e.value)


# ============================================================ sibling: the style path


def test_a_transposed_style_is_refused_rather_than_regrouped():
    """A ``(style_dim, B)`` style is refused by shape; its row count alone would pass.

    The square case (batch == style_dim) is ambiguous, so the rule is "the last axis is style_dim".
    """
    space = TinyEnv(num_envs=1, device=DEV).variation_space  # style_dim 3
    transposed = np.arange(15, dtype=np.float32).reshape(3, 5)  # meant to be (5, 3)
    assert transposed.size == 5 * 3, "the fixture must hold exactly 5 * 3 values"
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 5, device=DEV, values=transposed)
    assert "3 number(s) wide" in str(e.value) and "(3, 5)" in str(e.value)


def test_a_style_whose_last_axis_is_the_wrong_width_is_refused():
    space = TinyEnv(num_envs=1, device=DEV).variation_space
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 4, device=DEV, values=np.zeros((4, 2), np.float32))
    msg = str(e.value)
    assert "3 number(s) wide" in msg and "2" in msg, msg
    assert "style_names" in msg, f"the message does not say where the columns are listed: {msg}"


def test_a_style_with_the_wrong_ROW_count_is_still_refused_by_name():
    space = TinyEnv(num_envs=1, device=DEV).variation_space
    with pytest.raises(ValueError, match="style has 5 rows but 4 environment"):
        StyleTensor.from_space(space, 4, device=DEV, values=np.zeros((5, 3), np.float32))


def test_a_transposed_per_environment_variation_value_is_refused():
    """``variation={"block.color": (3, B)}`` is refused, and the message suggests the transpose."""
    space = TinyEnv(num_envs=1, device=DEV).variation_space
    with pytest.raises(ValueError) as e:
        StyleTensor.from_space(space, 8, device=DEV, overrides={"block.color": np.zeros((3, 8), np.float32)})
    msg = str(e.value)
    assert "transpose" in msg and "(8, 3)" in msg and "(3, 8)" in msg, msg


def test_an_unknown_variation_key_is_refused_rather_than_ignored():
    space = TinyEnv(num_envs=1, device=DEV).variation_space
    with pytest.raises(KeyError, match="unknown style path"):
        StyleTensor.from_space(space, 2, device=DEV, overrides={"blcok.color": [1, 2, 3]})


def test_an_unknown_path_in_sample_style_is_refused_rather_than_randomising_nothing():
    """A typo must not silently yield the default appearance for every environment."""
    env = _env(4)
    with pytest.raises(KeyError) as e:
        env.sample_style(4, paths=["block.colour"])  # British spelling, silent no-op
    msg = str(e.value)
    assert "block.colour" in msg and "block.color" in msg, f"neither the typo nor the fix: {msg}"


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
    """The corpus width comes from the env, not the input, so a frame stack is rejected at construction."""
    with pytest.raises(ValueError) as e:
        StateDataset(_env(1), torch.zeros((4, 8, 8, 3), dtype=torch.uint8), device=DEV)
    assert "FRAMES" in str(e.value) and "StateDataset" in str(e.value)


def test_a_state_dataset_refuses_the_wrong_state_width():
    with pytest.raises(ValueError, match="state_dim 5, got 2"):
        StateDataset(_env(1), np.zeros((10, 2), np.float32), device=DEV)


def test_a_state_dataset_still_flattens_a_leading_trajectory_axis():
    ds = StateDataset(_env(1), np.stack([id_states(4), id_states(4, first=10)]), device=DEV)
    assert len(ds) == 8


def test_a_trajectory_dataset_refuses_a_rank_three_frame_for_a_three_wide_env():
    """``(H, W, 3)`` is rank 3 and would read as H trajectories of W 3-wide states."""
    with pytest.raises(ValueError) as e:
        TrajectoryDataset(_env(1, cls=ThreeWideEnv), torch.zeros((6, 8, 3), dtype=torch.uint8), device=DEV)
    assert "FRAMES" in str(e.value), str(e.value)


def test_a_trajectory_dataset_refuses_a_transposed_style():
    st = np.stack([id_states(4), id_states(4, first=10)])
    with pytest.raises(ValueError) as e:
        TrajectoryDataset(_env(1), st, style=np.zeros((3, 2), np.float32), device=DEV)
    assert "3 number(s) wide" in str(e.value)


def test_a_trajectory_dataset_still_accepts_a_dense_corpus():
    ds = TrajectoryDataset(_env(1), np.stack([id_states(4), id_states(4, first=10)]), window=2, device=DEV)
    assert ds.n_trajectories == 2 and len(ds) == 6


# ============================================================ sibling: out= buffers


def test_a_strided_out_buffer_is_refused_by_name():
    """A strided ``out`` would scatter frames across the caller's neighboring slices."""
    buf = torch.zeros((4, 3, 8, 8, 3), dtype=torch.uint8)
    with pytest.raises(ValueError) as e:
        require_contiguous_out("test", buf[:, 1])
    msg = str(e.value)
    assert "contiguous" in msg and "(B, T, H, W, 3)" in msg, msg


def test_a_float_out_buffer_is_refused_for_every_env_not_just_the_rasterizer_ones():
    """``require_contiguous_out`` refuses a float ``out`` for every renderer, not only the rasterizers.

    A ``view``/``reshape`` into a float buffer would otherwise succeed and write 0..255 floats.
    """
    with pytest.raises(ValueError) as e:
        require_contiguous_out("test", torch.zeros((4, 8, 8, 3), dtype=torch.float32))
    assert "uint8" in str(e.value)
    require_contiguous_out("test", torch.zeros((4, 8, 8, 3), dtype=torch.uint8))  # the good case
    require_contiguous_out("test", None)  # and no buffer


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
    """``check_states`` ignores the env batch size, so ``render(states)`` can take any batch."""
    env = _env(2)
    out = check_states(env, id_states(7))
    assert out.shape == (7, 5) and out.dtype == torch.float32


def test_check_states_can_validate_without_moving_a_corpus_to_the_device():
    """``move=False`` is what lets ``render_states`` keep its bounded-peak-memory contract."""
    env = _env(2)
    big = torch.as_tensor(id_states(100))
    out = check_states(env, big, "t", leading=True, move=False)
    assert out.device.type == "cpu" and out.dtype == torch.float32
