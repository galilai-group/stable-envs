"""The recommended data path: the caller's own ``DataLoader`` over states, rendering in the step::

    for (states,) in DataLoader(TensorDataset(states), batch_size=256, shuffle=True):
        frames = env._set_render_state(states)._render()

Checked here: a plain ``DataLoader`` with ``drop_last=False`` runs to the end of its epoch, the datasets
expose no ``loader()`` factory, and the library datasets work under a caller-built loader (with
``collate_fn=collate_batched``, or with a ``BatchSampler`` and no argument at all). CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_input_validation import TinyEnv, _ids, _states  # noqa: E402  (same directory)
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

from stable_envs.testing._data import StateDataset, TrajectoryDataset, collate_batched

DEV = "cpu"


def _env(num_envs: int = 8) -> TinyEnv:
    return TinyEnv(num_envs=num_envs, device=DEV)


# ------------------------------------------------------------------ the recommended idiom


def test_a_plain_dataloader_with_drop_last_false_runs_to_the_end_of_its_epoch():
    """With torch's defaults every row reaches the renderer exactly once, including a short last batch."""
    env = _env(4)
    n = 10
    ds = TensorDataset(torch.as_tensor(_states(n)))
    seen, sizes = [], []
    for (states,) in DataLoader(ds, batch_size=4, shuffle=False, drop_last=False):
        frames = env._set_render_state(states)._render()
        sizes.append(int(frames.shape[0]))
        seen += _ids(frames)
    assert sizes == [4, 4, 2], "the short final batch was dropped, padded, or split"
    assert seen == list(range(n))


def test_get_state_is_what_the_step_should_use_for_its_target():
    """``get_state`` is on the env's device and trimmed to the rows rendered."""
    env = _env(4)
    env._set_render_state(_states(2))
    assert env._get_render_state().shape == (2, 5)
    assert env._get_render_state().device.type == DEV
    assert env._render().shape[0] == env._get_render_state().shape[0]


def test_states_that_are_already_on_the_env_s_device_are_accepted_unchanged():
    env = _env(4)
    s = torch.as_tensor(_states(4)).to(DEV)
    assert _ids(env._set_render_state(s)._render()) == [0, 1, 2, 3]


# ------------------------------------------------------------------ no loader factory


def test_the_datasets_no_longer_ship_a_dataloader_factory():
    """``StateDataset`` and ``TrajectoryDataset`` have no ``loader()``; callers build their own ``DataLoader``."""
    ds = StateDataset(_env(1), _states(6), device=DEV)
    assert not hasattr(ds, "loader"), "StateDataset must not expose a loader() factory"
    assert not hasattr(TrajectoryDataset(_env(1), np.stack([_states(4)]), device=DEV), "loader")


# ------------------------------------------------------------------ the library datasets under a caller's loader


def test_a_batch_sampler_iterates_a_state_dataset_with_torch_s_own_collate():
    """``batch_size=None`` with a ``BatchSampler`` needs no collate argument.

    The ``DataLoader`` hands the dataset a whole index list, so ``ds[list]`` does one gather.
    """
    ds = StateDataset(_env(1), _states(10), device=DEV)
    sampler = BatchSampler(SequentialSampler(range(len(ds))), 4, drop_last=False)
    got = list(DataLoader(ds, batch_size=None, sampler=sampler))
    assert [int(b.shape[0]) for b in got] == [4, 4, 2]
    assert _ids(torch.cat(got)) == list(range(10))


def test_indexing_a_state_dataset_with_a_list_returns_the_assembled_batch():
    """``ds[list]`` returns the assembled batch; ``ds[int]`` returns one sample."""
    ds = StateDataset(_env(1), _states(10), device=DEV)
    assert _ids(ds[[7, 2, 2, 0]]) == [7, 2, 2, 0]
    assert ds[3].shape == (3, 4, 3), "an integer index must give one sample, not a batch of one"


def test_the_batched_collate_still_serves_the_rendering_datasets():
    """``collate_fn=collate_batched`` iterates a rendering ``StateDataset``."""
    ds = StateDataset(_env(1), _states(8), device=DEV)
    got = list(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batched))
    assert len(got) == 2 and _ids(torch.cat(got)) == list(range(8))


def test_a_trajectory_dataset_under_a_caller_s_loader_keeps_its_time_axis():
    st = np.stack([_states(6), _states(6, first=100)])
    ds = TrajectoryDataset(_env(1), st, window=3, stride=1, device=DEV)
    batches = list(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batched))
    assert batches[0]["states"].shape == (4, 3, 5), "the window axis must survive the caller's loader"
