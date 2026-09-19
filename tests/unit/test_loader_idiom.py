"""The recommended data path: your own dataset and ``DataLoader`` over states, rendering in the step.

stable-envs supplies the renderer, not the data plumbing. Slicing a ``(N, state_dim)`` float32 array is
something torch and HuggingFace already do, so the idiom this package recommends imports nothing from
``stable_envs.data``::

    for (states,) in DataLoader(TensorDataset(states), batch_size=256, shuffle=True):
        frames = env.set_state(states).render()

Three properties make that work, and each is asserted below rather than asserted in a docstring:

  * a plain ``DataLoader`` with torch's default ``drop_last=False`` runs to the END of its epoch,
    because ``set_state`` renders a short final batch instead of refusing it;
  * ``StateDataset.loader()`` is gone — it wired one ``collate_fn`` argument and hid two decisions
    (batch size, workers) that belong to the caller;
  * the library's own datasets still work under a caller-built loader, with the one argument they
    genuinely need, and with a ``BatchSampler`` they need no argument at all.

CPU only. The env here is a real ``Env`` subclass on ``device="cpu"`` — see
``tests/unit/test_input_validation.py`` for why the guard tests in this area must not live in
``tests/gpu/``.

The measured numbers quoted in the docstrings are from one H200, PushT at 96 px, and are what decided
the design: the assembled-``__getitems__`` design was kept because per-item collation costs 43x on a
states-only ``TrajectoryDataset`` batch, and the ``BatchSampler`` recipe was documented because it
reaches the same throughput with torch's own collate.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

from stable_envs.data import StateDataset, TrajectoryDataset, collate_batched

from test_input_validation import TinyEnv, _ids, _states           # noqa: E402  (same directory)

DEV = "cpu"


def _env(num_envs: int = 8) -> TinyEnv:
    return TinyEnv(num_envs=num_envs, device=DEV)


# ------------------------------------------------------------------ the recommended idiom


def test_a_plain_dataloader_with_drop_last_false_runs_to_the_end_of_its_epoch():
    """The end-to-end shape of the recommended idiom, with torch's own defaults throughout.

    ``drop_last`` defaults to False, so this is what a user writes first, and every row of the dataset
    must reach the renderer exactly once. The failure this replaces was an exception on the final
    batch, whose suggested fix was to throw those rows away.
    """
    env = _env(4)
    n = 10
    ds = TensorDataset(torch.as_tensor(_states(n)))
    seen, sizes = [], []
    for (states,) in DataLoader(ds, batch_size=4, shuffle=False, drop_last=False):
        frames = env.set_state(states).render()
        sizes.append(int(frames.shape[0]))
        seen += _ids(frames)
    assert sizes == [4, 4, 2], "the short final batch was dropped, padded, or split"
    assert seen == list(range(n))


def test_get_state_is_what_the_step_should_use_for_its_target():
    """It is on the env's device and already trimmed, so a target cannot drift out of step.

    Using the loop variable instead works until the last batch of an epoch, at which point the frames
    are 2 rows and the target is still 2 rows — but the moment anything pads or the env is reused, the
    two disagree silently. ``get_state`` cannot.
    """
    env = _env(4)
    env.set_state(_states(2))
    assert env.get_state().shape == (2, 5)
    assert env.get_state().device.type == DEV
    assert env.render().shape[0] == env.get_state().shape[0]


def test_states_that_are_already_on_the_env_s_device_are_accepted_unchanged():
    env = _env(4)
    s = torch.as_tensor(_states(4)).to(DEV)
    assert _ids(env.set_state(s).render()) == [0, 1, 2, 3]


# ------------------------------------------------------------------ the API that was dropped


def test_the_datasets_no_longer_ship_a_dataloader_factory():
    """``ds.loader()`` is gone. Kept as an assertion rather than deleted quietly, so that a re-added
    convenience has to argue with this test first."""
    ds = StateDataset(_env(1), _states(6), device=DEV)
    assert not hasattr(ds, "loader"), (
        "StateDataset.loader is back; the recommended idiom is the caller's own DataLoader, and the "
        "one argument it needs (collate_fn=collate_batched, and only when the dataset renders) is "
        "documented in stable_envs/data.py")
    assert not hasattr(TrajectoryDataset(_env(1), np.stack([_states(4)]), device=DEV), "loader")


# ------------------------------------------------------------------ the library datasets under a caller's loader


def test_a_batch_sampler_iterates_a_state_dataset_with_torch_s_own_collate():
    """The fast bring-your-own recipe, and it needs NOTHING from this package to collate.

    ``batch_size=None`` + a ``BatchSampler`` makes the ``DataLoader`` hand the dataset a whole index
    list, so ``ds[list]`` does one gather and torch's ``default_convert`` passes the result through.
    Measured on PushT at B=1024: 2.886 ms/batch this way, 2.907 through ``__getitems__`` +
    ``collate_batched``, 5.637 for a per-item ``DataLoader``, and 2.774 for no loader at all.
    """
    ds = StateDataset(_env(1), _states(10), device=DEV)
    sampler = BatchSampler(SequentialSampler(range(len(ds))), 4, drop_last=False)
    got = list(DataLoader(ds, batch_size=None, sampler=sampler))
    assert [int(b.shape[0]) for b in got] == [4, 4, 2]
    assert _ids(torch.cat(got)) == list(range(10))


def test_indexing_a_state_dataset_with_a_list_returns_the_assembled_batch():
    """Which is what makes the recipe above work, and is asserted separately from the loader."""
    ds = StateDataset(_env(1), _states(10), device=DEV)
    assert _ids(ds[[7, 2, 2, 0]]) == [7, 2, 2, 0]
    assert ds[3].shape == (3, 4, 3), "an integer index must still give ONE sample, not a batch of one"


def test_the_batched_collate_still_serves_the_rendering_datasets():
    """``collate_batched`` stays: it is one argument, and the alternative was measured at 43x.

    (``TrajectoryDataset`` at B=1024: 8.741 ms/batch returning per-clip dicts against 0.202 ms
    returning the assembled dict, for a states-only batch where that IS the whole step.)
    """
    ds = StateDataset(_env(1), _states(8), device=DEV)
    got = list(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batched))
    assert len(got) == 2 and _ids(torch.cat(got)) == list(range(8))


def test_a_trajectory_dataset_under_a_caller_s_loader_keeps_its_time_axis():
    st = np.stack([_states(6), _states(6, first=100)])
    ds = TrajectoryDataset(_env(1), st, window=3, stride=1, device=DEV)
    batches = list(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batched))
    assert batches[0]["states"].shape == (4, 3, 5), "the window axis must survive the caller's loader"


# ``compression`` — the one thing ``StateDataset`` is worth importing for once the loader is the
# caller's own — is asserted in ``tests/unit/test_data.py``, not here. That version is strictly
# stronger: it makes the same four checks AND pins the one that can be wrong in a flattering direction,
# that the frame shape is MEASURED rather than assumed to be ``size x size`` (highway-env draws 150x600
# and Craftax 112x144, so a ratio computed from ``size**2`` overstates the saving by up to 4x). A
# second, weaker copy here would be the thing that gets updated when the arithmetic changes.
