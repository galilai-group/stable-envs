"""``render_states``, ``StateDataset`` and ``TrajectoryDataset``: the indexing, not the pixels.

``tests/gpu/test_reproducibility.py`` already drives all three over the whole registry with real
renderers and asserts the thing that matters most — that a stored corpus regenerates bit-identical
frames, through the loader, through ``all_frames``, and through a HuggingFace save/load. What it cannot
do is reach the failures that are NOT about pixels, because they need a corpus that is deliberately
malformed and an env whose frames say which state and which style produced them:

  * a window straddling an episode boundary teaches a transition that never happened and is invisible
    in a loss curve;
  * ``repeat`` instead of ``repeat_interleave`` gives trajectory j's appearance to trajectory i's
    states and renders without complaint (this one is called out by name in ``data.py`` and had no
    direct test);
  * padded input whose padding gets indexed produces frames of a state the env never reached;
  * a style with the wrong number of rows used to broadcast rather than raise.

Every one of them produces a plausible tensor of the right shape, which is why they are asserted here
rather than left to a shape check somewhere.

THE STAND-IN ENV. Nothing below needs a renderer, so it uses one that RECORDS what it was asked to
draw and encodes the answer in the frame: pixel (0,0) carries ``state[0]`` and pixel (0,1) carries
``style[0]``. That turns "did trajectory 1's states get trajectory 1's appearance" into an assertion on
bytes rather than an appeal to reading the code. It is 3x5 rather than square on purpose — several real
envs are not square (highway-env draws 150x600, Craftax 112x144) and a compression ratio computed from
``size**2`` is wrong by up to 4x for them, in the flattering direction. The convention of standing in
for an env with a small recording class is the one ``tests/unit/test_benchmarks.py`` already uses.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.testing._data import StateDataset, TrajectoryDataset, collate_batched, render_states

#: The frame is uint8, so the identity a pixel can carry is taken modulo this. A prime, so that
#: "state 3 got style 3" cannot come out true by a coincidence with the batch size — and any fixture
#: value a test then asserts on has to stay BELOW it, or the expectation has to wrap too.
MOD = 251


class RecordingEnv:
    """A StableEnv-shaped stand-in that draws WHICH state and WHICH style it was handed.

    Deliberately not a ``StableEnv`` subclass: the base class builds a variation space, a state buffer
    and a StyleTensor on a device, none of which this file is about, and inheriting them would make a
    CPU-only test depend on the parts of the stack it exists to run without.
    """

    state_dim = 2
    H, W = 3, 5
    state_names = ("x", "y")

    def __init__(self, style_dim: int = 2, size: int = 5):
        self.size = size
        self._style_dim = int(style_dim)
        self.device = torch.device("cpu")
        #: ``(states, style)`` for every render, so a test can assert on the PAIRING rather than on
        #: the pixels it produced.
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    @property
    def style_dim(self) -> int:
        return self._style_dim

    @property
    def style_names(self) -> list[str]:
        return [f"k{i}" for i in range(self._style_dim)]

    def default_style(self) -> np.ndarray:
        return np.zeros(self._style_dim, np.float32)

    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        s = torch.as_tensor(states, dtype=torch.float32).reshape(-1, self.state_dim)
        k = None
        if style is not None:
            k = torch.as_tensor(style, dtype=torch.float32).reshape(-1, self._style_dim)
        self.calls.append((s.clone(), None if k is None else k.clone()))
        n = s.shape[0]
        img = torch.zeros((n, self.H, self.W, 3), dtype=torch.uint8)
        img[:, 0, 0, :] = (s[:, 0].to(torch.int64) % MOD).to(torch.uint8).unsqueeze(-1)
        if k is not None and self._style_dim:
            img[:, 0, 1, :] = (k[:, 0].to(torch.int64) % MOD).to(torch.uint8).unsqueeze(-1)
        if out is not None:
            out.reshape(n, self.H, self.W, 3).copy_(img)
            return out
        return img


def _states(n, first=0):
    """``(n, 2)`` states whose first column is ``first, first+1, ...`` — a legible identity."""
    a = np.zeros((n, 2), np.float32)
    a[:, 0] = np.arange(first, first + n)
    a[:, 1] = 0.5
    return a


def _state_id(frames) -> np.ndarray:
    """The state identity the recording env drew, per frame."""
    f = frames.detach().cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    return f[..., 0, 0, 0].reshape(-1)


def _style_id(frames) -> np.ndarray:
    f = frames.detach().cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    return f[..., 0, 1, 0].reshape(-1)


# ------------------------------------------------------------------ collate_batched


def test_collate_is_the_identity_on_something_already_batched():
    """``__getitems__`` returns the assembled batch, and ``default_collate`` cannot take that.

    Measured on torch 2.11: ``default_collate`` treats a rendered ``(B,H,W,3)`` tensor as a LIST OF
    SAMPLES and hands it to ``torch.stack``, which raises "argument 'tensors' (position 1) must be
    tuple of Tensors, not Tensor". The obvious fix — return ``list(frames)`` so it re-stacks — copies
    the whole batch a second time (0.298 ms against the 2.535 ms render that produced it, i.e. 11.8%
    added to the step, for nothing).
    """
    t = torch.zeros(4, 3, 5, 3, dtype=torch.uint8)
    assert collate_batched(t) is t
    d = {"states": torch.zeros(4, 2), "traj": torch.zeros(4)}
    assert collate_batched(d) is d


def test_collate_still_handles_a_genuine_list_of_samples():
    """It has to stay safe to pass to ANY loader, including one over a dataset that is not ours."""
    out = collate_batched([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
    assert out.shape == (2, 2) and torch.equal(out[1], torch.tensor([3.0, 4.0]))


# ------------------------------------------------------------------ render_states


def test_leading_dimensions_survive_the_round_trip():
    """``(n_traj, T, D)`` in, ``(n_traj, T, H, W, 3)`` out, with the frames in the right cells.

    A reshape that transposes time into batch produces a stack of the right shape and the right total
    size holding the right frames in the wrong places, and no shape assertion anywhere would notice —
    which is why the identity of each frame is checked and not just the shape.
    """
    env = RecordingEnv()
    st = _states(12).reshape(3, 4, 2)
    got = render_states(env, st, device="cpu")
    assert got.shape == (3, 4, env.H, env.W, 3)
    assert _state_id(got).tolist() == list(range(12))


@pytest.mark.parametrize("chunk", [1, 3, 7, 12, 4096])
def test_the_chunk_bounds_memory_and_does_not_change_the_answer(chunk):
    """``chunk`` is where a stale ``out`` view or a mis-sliced style row lives, including on row 0."""
    env = RecordingEnv()
    st = _states(12)
    sty = np.stack([np.full(2, 100 + i, np.float32) for i in range(12)])
    got = render_states(env, st, style=sty, chunk=chunk, device="cpu")
    assert _state_id(got).tolist() == list(range(12))
    assert _style_id(got).tolist() == [100 + i for i in range(12)], (
        f"at chunk={chunk} the style rows are not the ones belonging to those states — a style sliced "
        f"with the CORPUS index instead of the chunk index gives every chunk after the first the "
        f"appearance of the rows before it"
    )


def test_a_single_style_row_is_shared_by_every_state():
    env = RecordingEnv()
    got = render_states(env, _states(6), style=np.full(2, 77, np.float32), chunk=4, device="cpu")
    assert _style_id(got).tolist() == [77] * 6


def test_a_style_that_is_neither_one_row_nor_one_per_state_is_refused():
    env = RecordingEnv()
    with pytest.raises(ValueError, match="style must have 1 or 6 rows"):
        render_states(env, _states(6), style=np.zeros((4, 2), np.float32), device="cpu")


def test_a_state_width_the_env_does_not_take_is_refused_by_name():
    env = RecordingEnv()
    with pytest.raises(ValueError, match="state_dim 2, got 5"):
        render_states(env, np.zeros((3, 5), np.float32), device="cpu")


def test_an_empty_corpus_is_an_error_rather_than_an_empty_tensor():
    """Returning ``(0,H,W,3)`` would need a render to learn H and W, so there is nothing to return.

    It is also almost always a caller bug — a filter that matched nothing, a slice off the end — and
    an empty tensor propagates silently into a training loop that then reports a loss of nan.
    """
    with pytest.raises(ValueError, match="no states to render"):
        render_states(RecordingEnv(), np.zeros((0, 2), np.float32), device="cpu")


def test_a_strided_out_buffer_is_refused_rather_than_silently_scattered():
    """The failure is silent in BOTH directions, which is why it is refused where the buffer arrives.

    ``reshape`` of a non-contiguous tensor returns a COPY, so every chunk would be rendered into a
    temporary and dropped; where it does return a view, the renderer writes through it linearly and the
    frames land across the neighbouring slices of the caller's buffer. ``buf[:, t]`` of a
    ``(B, T, H, W, 3)`` rollout buffer is exactly that shape, and it is the idiom the README recommends.
    """
    env = RecordingEnv()
    buf = torch.zeros((4, 3, env.H, env.W, 3), dtype=torch.uint8)
    view = buf[:, 1]
    assert not view.is_contiguous(), "the probe buffer must be strided for this to mean anything"
    with pytest.raises(ValueError, match="(?i)contiguous"):
        render_states(env, _states(4), out=view, device="cpu")
    assert int(buf.max()) == 0, "a refusal that scribbles first is not a refusal"


def test_a_numpy_out_buffer_is_filled_rather_than_crashed_on():
    """numpy >= 2 arrays carry a ``.device`` attribute, and the old duck-typed check died on it.

    ``getattr(flat, "device")`` is the string ``'cpu'`` for a numpy array, so ``.device.type`` raised
    "'str' object has no attribute 'type'" — for a caller doing the ordinary thing of rendering into an
    array they already own.
    """
    env = RecordingEnv()
    out = np.zeros((6, env.H, env.W, 3), np.uint8)
    got = render_states(env, _states(6), out=out, chunk=4, device="cpu")
    assert _state_id(got).tolist() == list(range(6))
    assert out[:, 0, 0, 0].tolist() == list(range(6))


# ------------------------------------------------------------------ StateDataset


def test_a_state_dataset_renders_the_rows_it_was_indexed_with():
    env = RecordingEnv()
    ds = StateDataset(env, _states(10), device="cpu")
    got = ds.__getitems__([7, 2, 2, 0])
    assert _state_id(got).tolist() == [7, 2, 2, 0], (
        "the batched hook rendered rows other than the ones asked for, or in another order"
    )
    assert _state_id(ds[[7, 2, 2, 0]]).tolist() == [7, 2, 2, 0], (
        "``ds[list]`` must reach the same batched hook — that is what lets a BatchSampler drive this "
        "dataset with torch's own collate and no argument from this package"
    )
    assert _state_id(ds[5]).tolist() == [5]
    assert ds[5].shape == (env.H, env.W, 3), (
        "an integer index must give ONE sample, not a batch of one; a caller who writes ds[i] and gets "
        "(1, H, W, 3) back stacks it into a time axis that is not there"
    )


def test_a_state_dataset_pairs_each_row_with_ITS_OWN_style():
    """Per-row appearance, indexed with the same indices as the states. The whole point of the pairing.

    Asserted through a shuffled, repeating index list rather than a contiguous slice, because a style
    indexed by ``arange(len(batch))`` — the plausible wrong version — agrees with the right answer on
    every contiguous batch starting at zero.
    """
    env = RecordingEnv()
    sty = np.stack([np.full(2, 200 + i, np.float32) for i in range(10)])
    ds = StateDataset(env, _states(10), style=sty, device="cpu")
    got = ds.__getitems__([9, 1, 4, 1])
    assert _state_id(got).tolist() == [9, 1, 4, 1]
    assert _style_id(got).tolist() == [209, 201, 204, 201]


def test_a_shared_style_row_is_not_indexed_at_all():
    env = RecordingEnv()
    ds = StateDataset(env, _states(10), style=np.full(2, 55, np.float32), device="cpu")
    assert _style_id(ds.__getitems__([3, 8])).tolist() == [55, 55]


def test_frames_false_hands_back_the_states_themselves():
    """The other supported idiom: the loader yields states and the training step renders them.

    The two differ in who owns the env — rendering in the dataset keeps it a drop-in for an image
    folder, rendering in the step lets the model choose resolution and style per batch — and this is
    the assertion that the second one is really a supported path rather than a sentence in a docstring.
    """
    env = RecordingEnv()
    ds = StateDataset(env, _states(6), frames=False, device="cpu")
    got = ds.__getitems__([4, 1])
    assert got.shape == (2, 2) and got[:, 0].tolist() == [4.0, 1.0]
    assert not env.calls, "frames=False rendered anyway"


def test_a_style_row_count_that_is_neither_is_refused():
    with pytest.raises(ValueError, match="style must have 1 or 10 rows"):
        StateDataset(RecordingEnv(), _states(10), style=np.zeros((4, 2), np.float32), device="cpu")


def test_compression_measures_the_frame_rather_than_assuming_it_is_square():
    """``size**2`` is wrong by up to 4x, in the flattering direction, for every non-square env."""
    env = RecordingEnv()  # 3x5, size 5
    ds = StateDataset(env, _states(100), device="cpu")
    c = ds.compression
    assert c["frame_shape"] == (3, 5, 3)
    assert c["frame_bytes"] == 100 * 3 * 5 * 3
    assert c["frame_bytes"] != 100 * env.size * env.size * 3, (
        "the frame size was assumed square; for this env that would overstate the saving by 1.7x"
    )
    assert c["state_bytes"] == 100 * 2 * 4 and c["stored_bytes"] == c["state_bytes"]
    assert c["ratio"] == pytest.approx(c["frame_bytes"] / c["stored_bytes"])


def test_compression_works_on_a_dataset_whose_style_is_per_row():
    """It used to RAISE here, and the cause is worth keeping: the probe renders ONE state.

    An env carrying a per-environment style refuses a one-row render, correctly and by design
    (``core/style.py``), so a shape probe that left the appearance to the env blew up on exactly the
    datasets that used domain randomisation. The dataset already knows the answer — it stores the style
    it renders with — so it passes row 0 of that, and the frame SHAPE is what is being measured, which
    no appearance changes.
    """
    env = RecordingEnv()
    sty = np.stack([np.full(2, 300 + i, np.float32) for i in range(20)])
    ds = StateDataset(env, _states(20), style=sty, device="cpu")
    assert ds.compression["frame_shape"] == (3, 5, 3)
    assert ds.compression["style_bytes"] == 20 * 2 * 4
    probe_style = env.calls[0][1]
    assert probe_style is not None and probe_style.shape[0] == 1, (
        "the shape probe must pass exactly one style row; leaving it to the env is what raised"
    )


def test_the_caller_s_own_dataloader_iterates_the_dataset_in_order():
    """``ds.loader()`` was dropped; the caller builds the ``DataLoader``, and it needs one argument.

    ``collate_fn=collate_batched``, because this dataset renders in ``__getitems__`` and hands back an
    already-assembled batch. ``num_workers=0`` belongs with it whenever the dataset renders — the
    frames come from a batched kernel on this process's GPU, so a worker would duplicate the CUDA
    context and ship back over IPC data that never needed to leave the device — but that is the
    caller's decision to make and torch already defaults to it.

    ``tests/unit/test_set_state_diagnostics.py`` covers the other two routes: the ``BatchSampler``
    recipe, which needs no collate argument at all, and the recommended idiom, where the states never
    go through this module.
    """
    from torch.utils.data import DataLoader

    ds = StateDataset(RecordingEnv(), _states(8), device="cpu")
    assert not hasattr(ds, "loader"), "the DataLoader factory is meant to be gone"
    batches = list(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batched))
    assert len(batches) == 2
    assert _state_id(batches[0]).tolist() == [0, 1, 2, 3]
    assert _state_id(batches[1]).tolist() == [4, 5, 6, 7]


# ------------------------------------------------------------------ TrajectoryDataset: the corpus shapes


def test_dense_input_becomes_flat_storage_with_one_length_per_trajectory():
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), device="cpu")
    assert ds.n_trajectories == 3 and ds.lengths.tolist() == [4, 4, 4]
    assert ds.states.shape == (12, 2) and ds.offsets.tolist() == [0, 4, 8]


def test_a_list_of_ragged_episodes_stays_ragged():
    """Ragged is the normal case, not a padded special case — which is why storage is flat."""
    eps = [_states(5, 0), _states(2, 100), _states(7, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, device="cpu")
    assert ds.lengths.tolist() == [5, 2, 7] and ds.states.shape == (14, 2)
    assert ds.states[5, 0].item() == 100 and ds.states[7, 0].item() == 200


def test_padded_input_keeps_only_the_live_prefix_so_padding_can_never_be_indexed():
    """A padded rectangle plus lengths is what most collectors write, and the padding is not a state.

    If it survived into storage, a clip could start inside it and the model would be trained on frames
    of a configuration the env never reached — which looks exactly like a normal frame.
    """
    dense = np.zeros((2, 5, 2), np.float32)
    dense[0, :, 0] = [0, 1, 2, 999, 999]
    dense[1, :, 0] = [10, 11, 12, 13, 14]
    ds = TrajectoryDataset(RecordingEnv(), dense, lengths=[3, 5], device="cpu")
    assert ds.states.shape == (8, 2)
    assert 999.0 not in ds.states[:, 0].tolist(), "the padding is still reachable"
    assert ds.states[:, 0].tolist() == [0, 1, 2, 10, 11, 12, 13, 14]


def test_flat_states_without_lengths_are_refused_rather_than_treated_as_one_episode():
    """Without boundaries a window silently spans two trajectories, which is the failure to prevent."""
    with pytest.raises(ValueError, match="need `lengths`"):
        TrajectoryDataset(RecordingEnv(), _states(10), device="cpu")


def test_lengths_that_do_not_account_for_every_row_are_refused():
    with pytest.raises(ValueError, match="lengths sum to 7 but states hold 10 rows"):
        TrajectoryDataset(RecordingEnv(), _states(10), lengths=[3, 4], device="cpu")


def test_a_state_width_the_env_does_not_take_is_refused_at_construction():
    with pytest.raises(ValueError, match="state_dim 2"):
        TrajectoryDataset(RecordingEnv(), np.zeros((2, 4, 5), np.float32), device="cpu")


# ------------------------------------------------------------------ TrajectoryDataset: the clip index


@pytest.mark.parametrize(
    "window,stride,frameskip,expect",
    [
        (1, 1, 1, 12),
        (2, 1, 1, 9),  # 3 clips per 4-step episode
        (4, 1, 1, 3),  # exactly one clip per episode
        (5, 1, 1, 0),  # longer than every episode: all dropped
        (2, 2, 1, 6),
        (2, 1, 2, 6),  # span 3, so 2 clips per episode
        (2, 1, 3, 3),  # span 4, so 1 clip per episode
    ],
)
def test_the_clip_count_follows_from_window_stride_and_frameskip(window, stride, frameskip, expect):
    """Three knobs that interact, and the span they imply is ``(window-1)*frameskip + 1``.

    Enumerated rather than derived in the test, because the assertion is that ``data.py``'s arithmetic
    matches the arithmetic somebody expects when they set these — an off-by-one here does not raise, it
    quietly drops the last transition of every episode or reads one past it.
    """
    ds = TrajectoryDataset(
        RecordingEnv(), _states(12).reshape(3, 4, 2), window=window, stride=stride, frameskip=frameskip, device="cpu"
    )
    assert len(ds) == expect


def test_a_window_never_straddles_an_episode_boundary_even_when_episodes_are_ragged():
    """THE property, checked on the rows a clip actually reads rather than on the count.

    ``_build_clips`` re-derives the last row of every clip and refuses one that runs past its own
    trajectory; this asserts the consequence independently, over ragged lengths where an off-by-one in
    the offsets is not masked by every episode being the same size.
    """
    eps = [_states(5, 0), _states(3, 100), _states(6, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, window=3, stride=1, device="cpu")
    assert len(ds) == 3 + 1 + 4, "the clip index is not the one this test was written against"
    for i in range(len(ds)):
        rows = ds[i]["states"][:, 0].tolist()
        # The episodes start at 0, 100 and 200, so a clip that crossed a boundary reads
        # e.g. [4.0, 100.0, 101.0] and fails the consecutiveness check by a hundred.
        assert rows == [rows[0] + k for k in range(3)], (
            f"clip {i} is not three consecutive steps of ONE episode: {rows}"
        )


def test_an_episode_shorter_than_the_window_is_dropped_and_counted():
    """Silently dropping them would make ``len(ds)`` a number nobody can reconcile with their corpus."""
    eps = [_states(5, 0), _states(2, 100), _states(6, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, window=4, device="cpu")
    assert ds.dropped_trajectories == 1
    assert ds.describe()["dropped_trajectories"] == 1
    assert {t for t, _ in ds.clip_indices} == {0, 2}


def test_clip_indices_are_relative_to_their_own_trajectory():
    """``(trajectory, start-within-trajectory)``, mirroring ``stable_worldmodel``'s name and meaning.

    Absolute row indices would be a different number that reads identically at a glance and is wrong
    for every trajectory after the first.
    """
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=2, device="cpu")
    assert ds.clip_indices[:3] == [(0, 0), (0, 1), (0, 2)]
    assert ds.clip_indices[3:6] == [(1, 0), (1, 1), (1, 2)]


def test_indexing_always_has_a_time_axis_even_at_window_one():
    """A shape that depends on a parameter's VALUE is where a downstream reshape transposes time.

    ``window=1`` returning ``(B, D)`` instead of ``(B, 1, D)`` is the tempting simplification, and the
    caller who then writes ``x.reshape(B, T, -1)`` gets a silent transposition rather than an error.
    """
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=1, device="cpu")
    assert ds.__getitems__([0, 1])["states"].shape == (2, 1, 2)
    assert ds[0]["states"].shape == (1, 2)


def test_a_list_index_and_a_scalar_index_agree():
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=2, device="cpu")
    one = ds[4]["states"]
    many = ds[[4]]["states"]
    assert torch.equal(one, many[0])


# ------------------------------------------------------------------ TrajectoryDataset: style is per trajectory


def test_style_is_expanded_over_a_trajectory_s_OWN_rows():
    """``repeat_interleave``, not ``repeat``. The defect ``data.py`` names and had no direct test.

    ``repeat`` tiles the style table across the corpus, so with 3 trajectories of 4 steps it gives row
    0 the styles ``[s0, s1, s2, s0]`` instead of ``[s0, s0, s0, s0]`` — trajectory j's appearance on
    trajectory i's states, rendered without complaint and wrong in a way no shape check sees. Asserted
    on ``per_row_style`` directly AND on the frames ``all_frames`` produces, because the two are
    separate opportunities to get it wrong.
    """
    env = RecordingEnv()
    sty = np.stack([np.full(2, 40 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=1, device="cpu")
    assert ds.per_row_style[:, 0].tolist() == [40] * 4 + [41] * 4 + [42] * 4
    got = ds.all_frames()
    assert _state_id(got).tolist() == list(range(12))
    assert _style_id(got).tolist() == [40] * 4 + [41] * 4 + [42] * 4


def test_style_travels_with_the_trajectory_through_ragged_episodes():
    """``repeat_interleave`` with a PER-ROW count, which a uniform repeat would get right by accident."""
    env = RecordingEnv()
    eps = [_states(2, 0), _states(5, 100), _states(3, 200)]
    sty = np.stack([np.full(2, 500 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, eps, style=sty, window=1, device="cpu")
    assert ds.per_row_style[:, 0].tolist() == [500] * 2 + [501] * 5 + [502] * 3


def test_a_single_style_row_is_given_to_every_trajectory():
    ds = TrajectoryDataset(
        RecordingEnv(), _states(12).reshape(3, 4, 2), style=np.full(2, 66, np.float32), window=1, device="cpu"
    )
    assert ds.style.shape == (3, 2) and ds.per_row_style[:, 0].tolist() == [66] * 12


def test_a_style_that_is_not_one_row_per_trajectory_is_refused():
    with pytest.raises(ValueError, match="1 or 3 rows \\(one per trajectory\\)"):
        TrajectoryDataset(
            RecordingEnv(), _states(12).reshape(3, 4, 2), style=np.zeros((12, 2), np.float32), device="cpu"
        )


def test_frames_true_renders_each_window_under_its_own_trajectory_s_style():
    """The per-clip version of the same expansion: ``B`` styles over ``B*window`` states."""
    env = RecordingEnv()
    sty = np.stack([np.full(2, 70 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=2, frames=True, device="cpu")
    out = ds.__getitems__([0, 5])  # clip 0 is traj 0, clip 5 is traj 1
    assert out["frames"].shape == (2, 2, 3, 5, 3)
    assert _state_id(out["frames"]).tolist() == [0, 1, 6, 7]
    assert _style_id(out["frames"]).tolist() == [70, 70, 71, 71]


def test_episode_returns_a_whole_trajectory_ignoring_the_window_index():
    """For evaluation and video, where a clip is the wrong unit."""
    env = RecordingEnv()
    sty = np.stack([np.full(2, 80 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=2, frames=True, device="cpu")
    ep = ds.episode(2)
    assert ep["states"].shape == (4, 2) and ep["states"][:, 0].tolist() == [8, 9, 10, 11]
    assert _style_id(ep["frames"]).tolist() == [82] * 4


def test_compression_counts_stored_rows_and_not_overlapping_clips():
    """Counting clips would multiply both sides by the overlap and inflate nothing but the apparent scale."""
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=2, device="cpu")
    c = ds.compression
    assert len(ds) == 9, "the overlap this test is about is not present"
    assert c["n_steps"] == 12 and c["frame_bytes"] == 12 * 3 * 5 * 3


# ------------------------------------------------------------------ HuggingFace


@pytest.fixture
def hf():
    return pytest.importorskip("datasets", reason="the HuggingFace round-trip needs `datasets`")


def test_a_corpus_round_trips_its_states_and_its_style_exactly(hf, tmp_path):
    """float32 in, float32 out, bit-identical — which is what makes the frames bit-identical.

    ``datasets`` hands values back as Python floats (C doubles) and ``np.asarray(..., float32)``
    narrows them again; widening then narrowing is exact, so the reload is the same numbers rather than
    nearly the same. The frames are compared too, because a corpus that round-trips its numbers but not
    its pixels is not a corpus.
    """
    env = RecordingEnv()
    sty = np.stack([np.full(2, 900 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=1, device="cpu")
    before = ds.all_frames().clone()

    back = TrajectoryDataset.from_hf(ds.to_hf(), env=env, window=1, device="cpu")
    assert torch.equal(back.states, ds.states)
    assert torch.equal(back.style, ds.style)
    assert torch.equal(back.all_frames(), before)

    p = str(tmp_path / "corpus")
    ds.save_to_disk(p)
    disk = TrajectoryDataset.load_from_disk(p, env=env, window=1, device="cpu")
    assert torch.equal(disk.all_frames(), before), "a corpus written and read back is not the corpus"


def test_ragged_episodes_survive_the_round_trip_as_rows(hf):
    """One row per TRAJECTORY, because the episode boundary is what a flat step table loses."""
    env = RecordingEnv()
    eps = [_states(2, 0), _states(5, 100), _states(3, 200)]
    ds = TrajectoryDataset(env, eps, window=1, device="cpu")
    back = TrajectoryDataset.from_hf(ds.to_hf(), env=env, window=1, device="cpu")
    assert back.lengths.tolist() == [2, 5, 3]
    assert torch.equal(back.states, ds.states)


def test_the_corpus_names_its_own_env_in_a_COLUMN_not_only_in_the_description(hf):
    """``info.description`` does not survive ``push_to_hub``; the column does.

    ``DatasetInfo._INCLUDED_INFO_IN_YAML`` (datasets 4.0.0) is
    ``[config_name, download_size, dataset_size, features, splits]`` — description is not in it — so a
    hub round trip comes back with an empty description and a corpus that has forgotten which renderer
    its states belong to. The column costs a few hundred bytes after parquet dictionary-encodes it.
    """
    from stable_envs.testing._data import HF_META_KEY

    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    hfds = ds.to_hf()
    assert HF_META_KEY in hfds.column_names
    import json

    meta = json.loads(hfds[HF_META_KEY][0])
    assert meta["env"] == "RecordingEnv" and meta["state_dim"] == 2
    assert meta["state_names"] == ["x", "y"] and meta["style_dim"] == 2


def test_a_corpus_that_names_no_env_is_refused_rather_than_guessed(hf):
    """Guessing from ``state_dim`` picks the wrong env the moment two envs share a width."""
    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    from stable_envs.testing._data import HF_META_KEY

    stripped = ds.to_hf().remove_columns([HF_META_KEY])
    stripped.info.description = ""
    with pytest.raises(ValueError, match="does not name its env"):
        TrajectoryDataset.from_hf(stripped, window=1, device="cpu")


def test_a_missing_datasets_package_explains_itself_instead_of_tracebacking(monkeypatch):
    """A missing optional dep is not a bug report, and the message says what to do AND what not to need.

    ``datasets`` IS installed here, so the absence is produced the only honest way — by making the
    import fail — rather than by replacing the function under test with one that raises the message the
    test then checks for, which would assert nothing about ``data.py`` at all.

    The alternative the message offers matters as much as the install line: a TrajectoryDataset is
    three arrays, and a user who does not want another dependency should be told that rather than left
    thinking the corpus format requires one.
    """
    import sys

    import stable_envs.testing._data as D

    monkeypatch.setitem(sys.modules, "datasets", None)  # `import datasets` now raises ImportError
    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    with pytest.raises(ImportError) as e:
        ds.to_hf()
    msg = str(e.value)
    assert "pip install datasets" in msg, "the message must name the install command"
    assert "np.savez" in msg, (
        "the message must offer the way out that needs no new dependency; without it a reader "
        "concludes the corpus format itself requires `datasets`"
    )
    assert D._require_datasets is not None
