"""FrameLoader hands out the frames of stored rows in order, a batch at a time, drawn in larger chunks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs import frames as frames_mod
from tests.helpers.tiny_env import TinyEnv, frame_ids, id_states


def _env(num_envs: int = 1) -> TinyEnv:
    return TinyEnv(num_envs=num_envs, device="cpu")


class _Counting(TinyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def render(self, x=None, **kwargs):
        self.calls.append(int(x.shape[0]))
        return super().render(x, **kwargs)


def test_frame_loader_is_public():
    assert snv.FrameLoader is frames_mod.FrameLoader
    assert "FrameLoader" in snv.__all__


@pytest.mark.parametrize("batch,chunk", [(1, None), (3, None), (3, 6), (3, 7), (4, 4), (10, None), (16, 4)])
def test_batches_come_in_order_with_a_short_last_batch(batch, chunk):
    env = _env()
    loader = snv.FrameLoader(env, id_states(10), batch, chunk_size=chunk)
    out = list(loader)
    assert [int(f.shape[0]) for f in out] == [min(batch, 10 - i) for i in range(0, 10, batch)]
    assert len(loader) == len(out)
    assert [i for f in out for i in frame_ids(f)] == list(range(10))
    assert all(f.dtype == torch.uint8 and f.shape[1:] == (3, 4, 3) for f in out)


def test_each_batch_has_the_bytes_of_rendering_that_batch():
    env = _env()
    rows = np.random.default_rng(0).uniform(0, 200, (23, 5)).astype(np.float32)
    for batch in (1, 4, 23):
        for got, start in zip(snv.FrameLoader(env, rows, batch, chunk_size=8), range(0, 23, batch), strict=True):
            assert torch.equal(got, env.render(rows[start : start + batch]))


def test_one_render_call_draws_many_batches():
    env = _Counting(num_envs=1, device="cpu")
    list(snv.FrameLoader(env, id_states(10), 2, chunk_size=6))
    assert env.calls == [6, 4]
    env.calls.clear()
    list(snv.FrameLoader(env, id_states(10), 2))
    assert env.calls == [10]


def test_a_chunk_is_a_multiple_of_the_batch():
    env = _Counting(num_envs=1, device="cpu")
    list(snv.FrameLoader(env, id_states(10), 3, chunk_size=7))
    assert env.calls == [6, 4]


def test_indices_pick_the_order_and_may_repeat():
    env = _env()
    order = [4, 0, 4, 9, 2]
    out = list(snv.FrameLoader(env, torch.as_tensor(id_states(10)), 2, indices=np.array(order), chunk_size=4))
    assert [i for f in out for i in frame_ids(f)] == order


def test_render_kwargs_pass_through():
    env = _env()
    got = next(iter(snv.FrameLoader(env, id_states(4), 4, variation={"block.color": (1, 2, 3)})))
    assert got[:, 0, 1].tolist() == [[1, 2, 3]] * 4


def test_a_per_environment_variation_keeps_one_batch_per_call():
    env = _Counting(num_envs=2, device="cpu")
    env.set_variation({"block.color": np.array([[1, 2, 3], [4, 5, 6]], np.float32)})
    loader = snv.FrameLoader(env, id_states(4), 2)
    assert loader.chunk_size == 2
    out = list(loader)
    assert env.calls == [2, 2]
    assert out[1][:, 0, 1].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_a_per_row_variation_keeps_one_batch_per_call():
    env = _env()
    rows = np.tile(np.float32([7, 8, 9]), (2, 1))
    assert snv.FrameLoader(env, id_states(4), 2, variation=rows).chunk_size == 2
    one = np.float32([[7, 8, 9]])
    assert snv.FrameLoader(env, id_states(4), 2, variation=one).chunk_size == 4


def test_chunk_rows_fits_memory_and_stays_a_multiple_of_the_batch(monkeypatch):
    env = _env()
    h, w = env.frame_spec.shape[:2]
    frame = h * w * 3
    per_row = 2 * frame + max(16 * h * w * env.ss**2, frames_mod.MIN_ROW_WORK)
    monkeypatch.setattr(frames_mod, "_free_bytes", lambda device: 4 * per_row * 50)
    assert frames_mod.chunk_rows(env, 8, 1000) == 48
    assert frames_mod.chunk_rows(env, 8, 20) == 20
    assert frames_mod.chunk_rows(env, 8, 52) == 48
    assert frames_mod.chunk_rows(env, 8, 5) == 5
    monkeypatch.setattr(frames_mod, "_free_bytes", lambda device: 0)
    assert frames_mod.chunk_rows(env, 8, 1000) == 8
    monkeypatch.setattr(frames_mod, "_free_bytes", lambda device: 1 << 50)
    assert frames_mod.chunk_rows(env, 1, 10**9) == frames_mod.MAX_CHUNK_BYTES // frame


def test_chunk_rows_reads_size_and_supersample():
    env = _env()
    assert frames_mod.chunk_rows(env, 1, 10**9, size=8192, supersample=2) <= frames_mod.MAX_CHUNK_BYTES // (8192 * 8192 * 3)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 2, "chunk_size": 0}, ValueError),
        ({"batch_size": 2, "out": torch.empty(0)}, TypeError),
        ({"batch_size": 2, "indices": [[0]]}, ValueError),
    ],
)
def test_bad_arguments_raise(kwargs, error):
    with pytest.raises(error):
        snv.FrameLoader(_env(), id_states(4), **kwargs)


def test_rows_must_be_two_dimensional():
    with pytest.raises(ValueError):
        snv.FrameLoader(_env(), id_states(4).reshape(-1), 2)
