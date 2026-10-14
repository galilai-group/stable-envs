"""Indexing of ``render_states``, ``StateDataset`` and ``TrajectoryDataset``.

Covers malformed-corpus cases that would otherwise yield a tensor of the right shape: windows that
straddle episodes, styles paired with the wrong trajectory, indexed padding, and style row counts.

The stand-in env writes ``state[0]`` into pixel (0,0) and ``style[0]`` into pixel (0,1), so pairing is
asserted on bytes. Its frame is 3x5 so that square-frame assumptions are caught.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.testing._data import StateDataset, TrajectoryDataset, collate_batched, render_states

#: Pixel identities are taken modulo this prime; fixture values asserted on must stay below it.
MOD = 251


class RecordingEnv:
    """A StableEnv-shaped stand-in that draws which state and which style it was handed.

    Not a ``StableEnv`` subclass, so the tests do not depend on variation spaces or device buffers.
    """

    state_dim = 2
    H, W = 3, 5
    state_names = ("x", "y")

    def __init__(self, style_dim: int = 2, size: int = 5):
        self.size = size
        self._style_dim = int(style_dim)
        self.device = torch.device("cpu")
        #: ``(states, style)`` for every render call.
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
    """``(n, 2)`` states whose first column is ``first, first+1, ...``."""
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
    """``collate_batched`` passes an already-assembled batch through without copying.

    ``default_collate`` treats a ``(B,H,W,3)`` tensor as a list of samples and fails in ``torch.stack``.
    """
    t = torch.zeros(4, 3, 5, 3, dtype=torch.uint8)
    assert collate_batched(t) is t
    d = {"states": torch.zeros(4, 2), "traj": torch.zeros(4)}
    assert collate_batched(d) is d


def test_collate_still_handles_a_genuine_list_of_samples():
    """A list of samples is still stacked, so the collate works with any dataset."""
    out = collate_batched([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
    assert out.shape == (2, 2) and torch.equal(out[1], torch.tensor([3.0, 4.0]))


# ------------------------------------------------------------------ render_states


def test_leading_dimensions_survive_the_round_trip():
    """``(n_traj, T, D)`` in, ``(n_traj, T, H, W, 3)`` out, with each frame in its own cell."""
    env = RecordingEnv()
    st = _states(12).reshape(3, 4, 2)
    got = render_states(env, st, device="cpu")
    assert got.shape == (3, 4, env.H, env.W, 3)
    assert _state_id(got).tolist() == list(range(12))


@pytest.mark.parametrize("chunk", [1, 3, 7, 12, 4096])
def test_the_chunk_bounds_memory_and_does_not_change_the_answer(chunk):
    """Chunking does not change which state or style row each frame gets."""
    env = RecordingEnv()
    st = _states(12)
    sty = np.stack([np.full(2, 100 + i, np.float32) for i in range(12)])
    got = render_states(env, st, style=sty, chunk=chunk, device="cpu")
    assert _state_id(got).tolist() == list(range(12))
    assert _style_id(got).tolist() == [100 + i for i in range(12)], (
        f"at chunk={chunk} the style rows do not match their states"
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
    """An empty corpus raises: H and W are unknown without a render."""
    with pytest.raises(ValueError, match="no states to render"):
        render_states(RecordingEnv(), np.zeros((0, 2), np.float32), device="cpu")


def test_a_strided_out_buffer_is_refused_rather_than_silently_scattered():
    """A non-contiguous ``out`` (e.g. ``buf[:, t]``) is refused before anything is written.

    ``reshape`` would either copy (dropping the frames) or write linearly across neighbouring slices.
    """
    env = RecordingEnv()
    buf = torch.zeros((4, 3, env.H, env.W, 3), dtype=torch.uint8)
    view = buf[:, 1]
    assert not view.is_contiguous(), "the probe buffer is not strided"
    with pytest.raises(ValueError, match="(?i)contiguous"):
        render_states(env, _states(4), out=view, device="cpu")
    assert int(buf.max()) == 0, "the refused buffer was written to"


def test_a_numpy_out_buffer_is_filled_rather_than_crashed_on():
    """numpy >= 2 arrays carry a string ``.device``, which must not be read as a torch device."""
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
    assert _state_id(got).tolist() == [7, 2, 2, 0], "__getitems__ rendered the wrong rows or order"
    assert _state_id(ds[[7, 2, 2, 0]]).tolist() == [7, 2, 2, 0], "ds[list] does not match __getitems__"
    assert _state_id(ds[5]).tolist() == [5]
    assert ds[5].shape == (env.H, env.W, 3), "an integer index must give one sample, not a batch of one"


def test_a_state_dataset_pairs_each_row_with_ITS_OWN_style():
    """Per-row style is indexed with the same indices as the states.

    A shuffled, repeating index list is used because ``arange(len(batch))`` would pass on a contiguous
    batch starting at zero.
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
    """With ``frames=False`` the dataset yields states and renders nothing."""
    env = RecordingEnv()
    ds = StateDataset(env, _states(6), frames=False, device="cpu")
    got = ds.__getitems__([4, 1])
    assert got.shape == (2, 2) and got[:, 0].tolist() == [4.0, 1.0]
    assert not env.calls, "frames=False rendered anyway"


def test_a_style_row_count_that_is_neither_is_refused():
    with pytest.raises(ValueError, match="style must have 1 or 10 rows"):
        StateDataset(RecordingEnv(), _states(10), style=np.zeros((4, 2), np.float32), device="cpu")


def test_compression_measures_the_frame_rather_than_assuming_it_is_square():
    """Frame bytes come from the rendered frame shape, not ``size**2``."""
    env = RecordingEnv()  # 3x5, size 5
    ds = StateDataset(env, _states(100), device="cpu")
    c = ds.compression
    assert c["frame_shape"] == (3, 5, 3)
    assert c["frame_bytes"] == 100 * 3 * 5 * 3
    assert c["frame_bytes"] != 100 * env.size * env.size * 3, "the frame size was assumed square"
    assert c["state_bytes"] == 100 * 2 * 4 and c["stored_bytes"] == c["state_bytes"]
    assert c["ratio"] == pytest.approx(c["frame_bytes"] / c["stored_bytes"])


def test_compression_works_on_a_dataset_whose_style_is_per_row():
    """The shape probe renders one state with exactly one stored style row."""
    env = RecordingEnv()
    sty = np.stack([np.full(2, 300 + i, np.float32) for i in range(20)])
    ds = StateDataset(env, _states(20), style=sty, device="cpu")
    assert ds.compression["frame_shape"] == (3, 5, 3)
    assert ds.compression["style_bytes"] == 20 * 2 * 4
    probe_style = env.calls[0][1]
    assert probe_style is not None and probe_style.shape[0] == 1, "the shape probe must pass one style row"


def test_the_caller_s_own_dataloader_iterates_the_dataset_in_order():
    """A plain ``DataLoader`` with ``collate_fn=collate_batched`` iterates the dataset in order.

    ``tests/unit/test_loader_idiom.py`` covers the absent ``loader()`` factory, the ``BatchSampler``
    route and the states-only idiom.
    """
    from torch.utils.data import DataLoader

    ds = StateDataset(RecordingEnv(), _states(8), device="cpu")
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
    """Ragged episodes are stored flat with one length each."""
    eps = [_states(5, 0), _states(2, 100), _states(7, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, device="cpu")
    assert ds.lengths.tolist() == [5, 2, 7] and ds.states.shape == (14, 2)
    assert ds.states[5, 0].item() == 100 and ds.states[7, 0].item() == 200


def test_padded_input_keeps_only_the_live_prefix_so_padding_can_never_be_indexed():
    """Padded input with ``lengths`` keeps only each trajectory's live prefix."""
    dense = np.zeros((2, 5, 2), np.float32)
    dense[0, :, 0] = [0, 1, 2, 999, 999]
    dense[1, :, 0] = [10, 11, 12, 13, 14]
    ds = TrajectoryDataset(RecordingEnv(), dense, lengths=[3, 5], device="cpu")
    assert ds.states.shape == (8, 2)
    assert 999.0 not in ds.states[:, 0].tolist(), "the padding is still reachable"
    assert ds.states[:, 0].tolist() == [0, 1, 2, 10, 11, 12, 13, 14]


def test_flat_states_without_lengths_are_refused_rather_than_treated_as_one_episode():
    """Flat states need ``lengths``; otherwise windows could span trajectories."""
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
    """Clip counts for window, stride and frameskip; a clip spans ``(window-1)*frameskip + 1`` steps."""
    ds = TrajectoryDataset(
        RecordingEnv(), _states(12).reshape(3, 4, 2), window=window, stride=stride, frameskip=frameskip, device="cpu"
    )
    assert len(ds) == expect


def test_a_window_never_straddles_an_episode_boundary_even_when_episodes_are_ragged():
    """Every clip reads consecutive rows of one episode, checked over ragged lengths."""
    eps = [_states(5, 0), _states(3, 100), _states(6, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, window=3, stride=1, device="cpu")
    assert len(ds) == 3 + 1 + 4, "unexpected clip count"
    for i in range(len(ds)):
        rows = ds[i]["states"][:, 0].tolist()
        # The episodes start at 0, 100 and 200, so a clip that crossed a boundary reads
        # e.g. [4.0, 100.0, 101.0] and fails the consecutiveness check by a hundred.
        assert rows == [rows[0] + k for k in range(3)], (
            f"clip {i} is not three consecutive steps of one episode: {rows}"
        )


def test_an_episode_shorter_than_the_window_is_dropped_and_counted():
    """Episodes shorter than the window are dropped and reported in ``dropped_trajectories``."""
    eps = [_states(5, 0), _states(2, 100), _states(6, 200)]
    ds = TrajectoryDataset(RecordingEnv(), eps, window=4, device="cpu")
    assert ds.dropped_trajectories == 1
    assert ds.describe()["dropped_trajectories"] == 1
    assert {t for t, _ in ds.clip_indices} == {0, 2}


def test_clip_indices_are_relative_to_their_own_trajectory():
    """``clip_indices`` are ``(trajectory, start-within-trajectory)``, as in ``stable_worldmodel``."""
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=2, device="cpu")
    assert ds.clip_indices[:3] == [(0, 0), (0, 1), (0, 2)]
    assert ds.clip_indices[3:6] == [(1, 0), (1, 1), (1, 2)]


def test_indexing_always_has_a_time_axis_even_at_window_one():
    """``window=1`` still returns a time axis: ``(B, 1, D)``, not ``(B, D)``."""
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
    """Each trajectory's style is expanded over its own rows (``repeat_interleave``, not ``repeat``).

    Checked on ``per_row_style`` and on the frames ``all_frames`` produces.
    """
    env = RecordingEnv()
    sty = np.stack([np.full(2, 40 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=1, device="cpu")
    assert ds.per_row_style[:, 0].tolist() == [40] * 4 + [41] * 4 + [42] * 4
    got = ds.all_frames()
    assert _state_id(got).tolist() == list(range(12))
    assert _style_id(got).tolist() == [40] * 4 + [41] * 4 + [42] * 4


def test_style_travels_with_the_trajectory_through_ragged_episodes():
    """Style expansion uses per-trajectory lengths on ragged episodes."""
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
    """Each clip is rendered with its own trajectory's style: ``B`` styles over ``B*window`` states."""
    env = RecordingEnv()
    sty = np.stack([np.full(2, 70 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=2, frames=True, device="cpu")
    out = ds.__getitems__([0, 5])  # clip 0 is traj 0, clip 5 is traj 1
    assert out["frames"].shape == (2, 2, 3, 5, 3)
    assert _state_id(out["frames"]).tolist() == [0, 1, 6, 7]
    assert _style_id(out["frames"]).tolist() == [70, 70, 71, 71]


def test_episode_returns_a_whole_trajectory_ignoring_the_window_index():
    """``episode`` returns a whole trajectory with its frames."""
    env = RecordingEnv()
    sty = np.stack([np.full(2, 80 + i, np.float32) for i in range(3)])
    ds = TrajectoryDataset(env, _states(12).reshape(3, 4, 2), style=sty, window=2, frames=True, device="cpu")
    ep = ds.episode(2)
    assert ep["states"].shape == (4, 2) and ep["states"][:, 0].tolist() == [8, 9, 10, 11]
    assert _style_id(ep["frames"]).tolist() == [82] * 4


def test_compression_counts_stored_rows_and_not_overlapping_clips():
    """Compression counts stored rows, not overlapping clips."""
    ds = TrajectoryDataset(RecordingEnv(), _states(12).reshape(3, 4, 2), window=2, device="cpu")
    c = ds.compression
    assert len(ds) == 9, "the clips do not overlap"
    assert c["n_steps"] == 12 and c["frame_bytes"] == 12 * 3 * 5 * 3


# ------------------------------------------------------------------ HuggingFace


@pytest.fixture
def hf():
    return pytest.importorskip("datasets", reason="the HuggingFace round-trip needs `datasets`")


def test_a_corpus_round_trips_its_states_and_its_style_exactly(hf, tmp_path):
    """States, style and frames round-trip bit-identically through HuggingFace and disk.

    ``datasets`` returns Python floats; narrowing them back to float32 is exact.
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
    assert torch.equal(disk.all_frames(), before), "frames changed after save_to_disk/load_from_disk"


def test_ragged_episodes_survive_the_round_trip_as_rows(hf):
    """Ragged episodes round-trip as one row per trajectory, keeping their lengths."""
    env = RecordingEnv()
    eps = [_states(2, 0), _states(5, 100), _states(3, 200)]
    ds = TrajectoryDataset(env, eps, window=1, device="cpu")
    back = TrajectoryDataset.from_hf(ds.to_hf(), env=env, window=1, device="cpu")
    assert back.lengths.tolist() == [2, 5, 3]
    assert torch.equal(back.states, ds.states)


def test_the_corpus_names_its_own_env_in_a_COLUMN_not_only_in_the_description(hf):
    """The env metadata is stored in a column, since ``info.description`` does not survive ``push_to_hub``."""
    from stable_envs.testing._data import HF_META_KEY

    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    hfds = ds.to_hf()
    assert HF_META_KEY in hfds.column_names
    import json

    meta = json.loads(hfds[HF_META_KEY][0])
    assert meta["env"] == "RecordingEnv" and meta["state_dim"] == 2
    assert meta["state_names"] == ["x", "y"] and meta["style_dim"] == 2


def test_a_corpus_that_names_no_env_is_refused_rather_than_guessed(hf):
    """A corpus without env metadata is refused rather than matched by ``state_dim``."""
    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    from stable_envs.testing._data import HF_META_KEY

    stripped = ds.to_hf().remove_columns([HF_META_KEY])
    stripped.info.description = ""
    with pytest.raises(ValueError, match="does not name its env"):
        TrajectoryDataset.from_hf(stripped, window=1, device="cpu")


def test_a_missing_datasets_package_explains_itself_instead_of_tracebacking(monkeypatch):
    """Without ``datasets`` the ImportError names the install command and the ``np.savez`` alternative.

    The absence is simulated by making ``import datasets`` fail.
    """
    import sys

    import stable_envs.testing._data as D

    monkeypatch.setitem(sys.modules, "datasets", None)  # `import datasets` now raises ImportError
    ds = TrajectoryDataset(RecordingEnv(), _states(8).reshape(2, 4, 2), window=1, device="cpu")
    with pytest.raises(ImportError) as e:
        ds.to_hf()
    msg = str(e.value)
    assert "pip install datasets" in msg, "the message must name the install command"
    assert "np.savez" in msg, "the message must offer the np.savez alternative"
    assert D._require_datasets is not None
