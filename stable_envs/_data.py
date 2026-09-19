"""Private corpus helpers used by certification and research tooling.

Dataset storage is not part of the public stable-envs API. Ordinary users keep compact states in
their own data stack and call ``env.render``. The helpers here remain because certification
exercises chunked rendering, trajectory boundaries, style serialization, and optional HuggingFace
round trips. They are private and may change without compatibility guarantees.

State datasets in, frames out. The entry point for "stop storing pixels".

This is what the repo is for. A trajectory corpus already contains the compact state at every step; the
frames stored beside it are redundant with those numbers, and regenerating them on the GPU is faster
than reading them back from disk. PushT is the extreme case — 5 float32 against a 96x96x3 frame is 20
bytes against 27,648, a 1382x shrink, and the frames come back at 3.2 M/s — but the same arithmetic
holds wherever the state-to-frame claim does.

Concretely, at 96px:

    stored frames   27,648 B/step, read at maybe 2 GB/s from a fast local disk  ->  ~72 k steps/s
    stored states       20 B/step, rendered on one H200                         ->  ~3.2 M steps/s

so the GPU path is ~45x faster than reading the frames it replaces, before counting the disk you did
not fill.

WHAT THIS MODULE IS AND IS NOT FOR
----------------------------------

The recommended way to train against these renderers does not use this module at all::

    for states in my_own_dataloader:                  # your dataset, your sampler, your workers
        frames = env.set_state(states).render()       # (B, H, W, 3) uint8, one launch

stable-envs supplies the renderer. Slicing a corpus of states is something torch and HuggingFace
already do well, and a library-owned ``DataLoader`` wrapper around a ``(N, state_dim)`` float32 array
earns nothing. So what is left here is the part that is genuinely about rendering or about episode
structure:

    render_states       a corpus of states -> frames, chunked so peak memory is the chunk, not N
    StateDataset        stores states, renders in __getitems__: a drop-in for an image folder, for
                        migrating an existing image pipeline without touching its training step
    TrajectoryDataset   (n_traj, T, state_dim) + per-trajectory style, yielding windows that never
                        straddle an episode boundary, with HuggingFace round-trips

Every entry point here validates what it is given before it reshapes it — see ``env.check_states``
for the rule and for the silent-garbage path it closes. ``StateDataset`` used to take its state width
from the INPUT (``reshape(-1, st.shape[-1])``), so any array at all was a valid corpus and a frame
stack became a dataset that either failed much later or did not fail at all; it now takes the width
from the env, which is the only thing that knows it. ``check_style`` applies the same shape-not-size
rule to the appearance half.

``TrajectoryDataset`` carries the style tensor because style is state (see ``core/style.py``): a corpus
that stores only the dynamic half can be re-rendered only under whatever appearance happens to be set
at render time, which is a coincidence rather than reproduction. The HuggingFace round-trip therefore
stores ``(states, style)`` together, and ``examples/hf_roundtrip.py`` asserts that frames rendered after
a save/load are **bit-identical** to the frames rendered before it — that assertion is the claim.

COLLATION, AND WHY THERE IS NO ``ds.loader()`` ANY MORE
------------------------------------------------------

Both datasets implement ``__getitems__``, the batched hook ``DataLoader`` prefers, and return the
ASSEMBLED batch from it. ``default_collate`` then receives something already collated and dies with
``stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor`` (measured, torch
2.11), so those two datasets need ``collate_fn=stable_envs.collate_batched``. There used to be a
``ds.loader()`` that wired it; it is gone, because it presented a one-line ``DataLoader`` argument as a
library API and hid the two decisions (batch size, workers) a caller should be making themselves.

Returning per-item samples instead — the obvious way to make ``default_collate`` work — was measured
and rejected. On one H200, one epoch's worth of PushT batches, states only, no render:

    TrajectoryDataset __getitems__ -> dict of batched tensors      0.202 ms/batch at B=1024
    TrajectoryDataset __getitems__ -> list of per-clip dicts       8.741 ms/batch at B=1024   (43x)

The list form pays B python dict constructions and B*4 tensor slices, then makes ``default_collate``
stack all of it back. For a states-only batch that is the entire cost of the step.

If you do bring your own loader — the recommended idiom — the same measurement says how to keep it
free. Indexing a dataset per item costs about 3.4 us per row of python, which is invisible at B=64 and
is 14.9 ms at B=4096. A ``BatchSampler`` removes it with no custom collate at all::

    from torch.utils.data import BatchSampler, DataLoader, RandomSampler
    sampler = BatchSampler(RandomSampler(range(len(states))), batch_size=1024, drop_last=False)
    loader = DataLoader(states_dataset, batch_size=None, sampler=sampler)   # ds[list] -> (B, D)

measured at 2.886 ms/batch against 2.907 for the assembled-``__getitems__`` path and 2.774 for no
loader at all, where a plain per-item ``DataLoader`` takes 5.637.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .env import Env, check_states, require_contiguous_out
from .registry import make

#: Written into the HuggingFace dataset description so a corpus says which renderer it belongs to.
#: Without it a reload has to be told the env by hand, and a corpus that does not name its own env is
#: not self-describing — the states alone do not say what they mean.
HF_META_KEY = "stable_envs"


def _states_tensor(x) -> torch.Tensor:
    """Anything array-like -> contiguous float32 CPU tensor, without a device round-trip surprise."""
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).contiguous()


def check_style(env, style, rows: int, owner: str, per: str = "state") -> torch.Tensor:
    """``style`` -> a contiguous ``(1 or rows, style_dim)`` float32 CPU tensor, or a real message.

    The same shape-not-size rule the state path uses, and for the same reason on the same quantity.
    ``reshape(-1, style_dim)`` succeeds on a transposed ``(style_dim, B)`` array whenever ``B`` equals
    ``style_dim``, and hands back rows that are mixtures of different environments' knobs; the row
    count check downstream then passes, because the count is right. "A per-environment appearance
    silently became something else" is the defect ``core/style.py`` exists to make unreachable, and it
    has shipped three times.
    """
    k = style if torch.is_tensor(style) else np.asarray(style)
    dim = int(env.style_dim)
    nd = int(getattr(k, "ndim", 0))
    if nd >= 2 and int(k.shape[-1]) != dim:
        raise ValueError(
            f"{owner}: style rows are {dim} number(s) wide (env.style_names) and the last axis of "
            f"this one is {int(k.shape[-1])} (shape {tuple(k.shape)}). Flattening it by size would "
            f"succeed on a transpose and give each row a mixture of other rows' knobs, so it is "
            f"refused. Pass ({rows}, {dim}) — one row per {per} — or ({dim},) to share one "
            f"appearance."
        )
    t = _states_tensor(style)
    if dim and t.numel() % dim:
        raise ValueError(
            f"{owner}: a flat style must be a whole number of {dim}-wide rows, and {t.numel()} is "
            f"not a multiple of {dim}."
        )
    t = t.reshape(-1, dim) if dim else t.reshape(-1, 0)
    if t.shape[0] not in (1, rows):
        raise ValueError(f"{owner}: style must have 1 or {rows} rows (one per {per}), got {t.shape[0]}")
    return t.contiguous()


def collate_batched(batch):
    """Collate for datasets whose ``__getitems__`` already returns the assembled batch.

    ``_MapDatasetFetcher`` calls ``dataset.__getitems__(indices)`` when it exists and then passes the
    result to ``collate_fn`` unchanged. Ours returns a rendered ``(B,H,W,3)`` tensor (or a dict of
    batched tensors), which ``default_collate`` treats as a *list of samples* and hands to
    ``torch.stack``, raising ``stack(): argument 'tensors' (position 1) must be tuple of Tensors, not
    Tensor``. The alternative — returning per-item samples so ``default_collate`` re-stacks them — is
    correct and was measured against this one: for ``TrajectoryDataset`` at B=1024 it is 8.741 ms/batch
    against 0.202, a 43x tax on a states-only step, because it pays B python dict constructions and
    B*4 tensor slices to undo a gather that had already happened. For ``StateDataset(frames=True)`` it
    is one more full copy of the pixels.

    So this is the identity on an already-batched object and falls back to ``default_collate`` for a
    genuine list of samples, which keeps it safe to pass to any loader::

        DataLoader(ds, batch_size=256, shuffle=True, collate_fn=stable_envs.collate_batched,
                   num_workers=0)

    ``num_workers=0`` belongs with it whenever the dataset renders (``StateDataset(frames=True)``,
    ``TrajectoryDataset(frames=True)``): the frames are produced by a batched kernel on this process's
    GPU, so a worker would build a second CUDA context and ship over IPC data that never needed to
    leave the device. A states-only dataset has no such constraint.
    """
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        return batch
    from torch.utils.data._utils.collate import default_collate

    return default_collate(batch)


@torch.no_grad()
def render_states(
    env: Env | str,
    states,
    size: int | None = None,
    chunk: int = 8192,
    variation: dict | None = None,
    style=None,
    out=None,
    device: str = "cuda",
    **env_kw,
):
    """``(...,state_dim)`` states -> ``(...,H,W,3)`` uint8 frames.

    ``env`` may be a live env or a registry name. Leading dims are preserved, so a ``(n_traj, T, D)``
    corpus comes back as ``(n_traj, T, H, W, 3)`` without the caller flattening and reshaping — the
    shape bookkeeping is exactly where an off-by-one silently transposes a dataset.

    ``style`` is a ``(style_dim,)`` row shared by every state, or one row per state after the leading
    dims are flattened. It is accepted here rather than being left to the caller because the frame is a
    function of ``(state, style)`` and dropping half of that pair is how a "reproduced" dataset quietly
    becomes a differently-coloured one.
    """
    # The env has to exist before the states can be checked (only it knows ``state_dim``), and the
    # states have to be sized before the env can be built at a sensible batch. So: a provisional row
    # count from the input's own shape, then the env, then the real check.
    probe_shape = tuple(getattr(states if torch.is_tensor(states) else np.asarray(states), "shape", ()))
    n_guess = int(np.prod(probe_shape[:-1])) if len(probe_shape) >= 1 else 1
    if isinstance(env, str):
        env = make(env, num_envs=max(1, min(chunk, n_guess or 1)), size=size, device=device, **env_kw)

    # ``move=False``: the corpus stays wherever the caller put it and is copied one chunk at a time,
    # which is the whole contract of this function. The NaN/inf scan therefore runs on the host for a
    # host corpus, i.e. for free.
    st = check_states(env, states, "render_states", leading=True, move=False)
    lead = tuple(st.shape[:-1])
    st = st.contiguous().reshape(-1, env.state_dim)
    N = st.shape[0]
    S = size or env.size

    sty = None
    if style is not None:
        sty = check_style(env, style, N, "render_states(style=)")

    probe = env.render(
        st[:1].to(device), size=S, variation=variation, style=None if sty is None else sty[:1].to(device)
    )
    H, W = int(probe.shape[1]), int(probe.shape[2])

    if out is None:
        out = torch.empty((N, H, W, 3), dtype=torch.uint8, device=device)
    # A strided destination cannot be filled here at all, and the failure is silent in BOTH directions:
    # ``reshape`` of a non-contiguous tensor returns a COPY, so every chunk would be written into a
    # temporary and dropped, and where it does return a view the renderer writes through it linearly
    # and scatters the frames. Refuse it once, where the buffer arrives.
    require_contiguous_out("render_states(out=)", out)
    flat = out.reshape(N, H, W, 3)
    # numpy>=2 arrays carry a ``.device`` attribute (the string 'cpu'), so ``getattr(flat, "device")``
    # is truthy for a numpy ``out=`` and the old check crashed with "'str' object has no attribute
    # 'type'". Ask torch, not duck typing.
    on_device = torch.is_tensor(flat) and flat.device.type == torch.device(device).type

    for i in range(0, N, chunk):
        s = st[i : i + chunk].to(device, non_blocking=True)
        k = None if sty is None else (sty if sty.shape[0] == 1 else sty[i : i + chunk]).to(device)
        if on_device:
            env.render(s, size=S, variation=variation, style=k, out=flat[i : i + chunk])
        else:
            flat[i : i + chunk] = env.render(s, size=S, variation=variation, style=k).cpu()
    return flat.reshape(*lead, H, W, 3) if lead else flat


class _RenderingDataset(torch.utils.data.Dataset):
    """Shared machinery: hold an env, render batches, report what storing states saved.

    Split out because ``StateDataset`` and ``TrajectoryDataset`` differ only in what they index — a flat
    bag of rows versus windows inside episodes — and duplicating the env plumbing is how the two drift
    into disagreeing about the frame shape or the style handling.
    """

    env: Env
    size: int
    variation: dict | None
    device: str

    def _init_env(self, env, size, variation, device, env_kw):
        self.env = make(env, num_envs=1, size=size, device=device, **env_kw) if isinstance(env, str) else env
        self.size = size or self.env.size
        self.variation = variation
        self.device = device
        self._frame_hw: tuple[int, int] | None = None

    @torch.no_grad()
    def render(self, states, style=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device. The call a training step makes."""
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.env.state_dim)
        k = None
        if style is not None:
            k = torch.as_tensor(style, device=self.device, dtype=torch.float32)
            if k.ndim >= 2 and int(k.shape[-1]) != self.env.style_dim:
                raise ValueError(
                    f"{type(self).__name__}.render: style rows are {self.env.style_dim} number(s) "
                    f"wide and the last axis of this one is {int(k.shape[-1])} (shape "
                    f"{tuple(k.shape)}). Flattening it by size succeeds on a transpose and gives each "
                    f"row a mixture of other rows' knobs, so it is refused."
                )
            k = k.reshape(-1, self.env.style_dim)
            if k.shape[0] != s.shape[0]:
                # One style row per window, states flattened over time: repeat_interleave, not repeat.
                # ``repeat`` would tile the styles across the batch and give trajectory j's appearance
                # to trajectory i's states, which renders without complaint.
                if s.shape[0] % k.shape[0]:
                    raise ValueError(f"style rows {k.shape[0]} do not divide state rows {s.shape[0]}")
                k = k.repeat_interleave(s.shape[0] // k.shape[0], dim=0)
        img = self.env.render(s, size=self.size, variation=self.variation, style=k)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def frame_shape(self) -> tuple[int, int, int]:
        """``(H,W,3)``, measured by rendering one frame rather than assumed to be ``size x size``.

        Several envs are not square — highway-env draws 150x600, Craftax 112x144 — so a compression
        ratio computed from ``size**2`` is wrong by up to 4x for them, in the flattering direction.

        The probe renders ONE state, and it must say which appearance to draw it under. Leaving that to
        the env is only safe while the env's own style is shared: an env carrying a per-environment
        style refuses a one-row render, correctly and by design (``core/style.py``), so
        ``ds.compression`` on a dataset built from an env that had a randomised appearance set on it
        raised instead of returning a shape. The dataset already knows the answer — it stores the style
        it renders with — so it passes row 0 of that, and falls back to the env's default appearance
        when it carries none. Either way the frame SHAPE is what is being measured, and no appearance
        changes it.
        """
        if self._frame_hw is None:
            style = getattr(self, "style", None)
            if style is not None:
                style = torch.as_tensor(style, device=self.device, dtype=torch.float32).reshape(
                    -1, self.env.style_dim
                )[:1]
            elif self.env.style_dim:
                style = torch.as_tensor(self.env.default_style(), device=self.device, dtype=torch.float32).reshape(
                    1, -1
                )
            with torch.no_grad():
                f = self.env.render(self._probe_state(), size=self.size, variation=self.variation, style=style)
            self._frame_hw = (int(f.shape[-3]), int(f.shape[-2]))
        return (*self._frame_hw, 3)

    def _probe_state(self) -> torch.Tensor:
        raise NotImplementedError


class StateDataset(_RenderingDataset):
    """A frame dataset that stores states and renders on access. A drop-in for an image dataset.

    THIS IS THE MIGRATION SHIM, NOT THE RECOMMENDED IDIOM. It exists so that a pipeline already built
    around an image folder can point at states instead without its training step changing: the object
    still yields ``(B, H, W, 3)`` uint8, so ``for frames in loader`` keeps working and the corpus on
    disk shrinks by three orders of magnitude. If you are writing the training step anyway, own the
    loader yourself and call ``env.set_state(states).render()`` in it — that is the idiom this repo
    recommends, it needs nothing from this module, and it lets the step pick resolution and style per
    batch instead of freezing them into a dataset.

    Rendering happens in ``__getitems__`` (the batched hook ``DataLoader`` >= 2 prefers) rather than per
    item: one launch for the whole batch is the entire point, and a per-item render would hand back the
    launch overhead this repo spent its effort removing. That makes the returned object already
    collated, so a ``DataLoader`` over it needs ``collate_fn=stable_envs.collate_batched`` and
    ``num_workers=0``; see ``collate_batched`` for the exact error ``default_collate`` raises otherwise
    and for what the per-item alternative was measured to cost.

    ``style`` is stored alongside the states, one row per state or one row shared, because the frame is
    a function of both.

    ``frames=False`` yields the raw states instead. It is a convenience for the case where the state
    corpus is already here — ``compression`` and the env are attached to it, and ``DistributedSampler``
    shards it exactly as it shards an image folder — but a plain ``torch.utils.data.TensorDataset`` over
    the same array does the same job, and neither one renders.

    ``ds[[i, j, k]]`` returns the assembled batch for those rows, which is what makes the
    ``BatchSampler`` recipe in this module's docstring work on it with the DEFAULT collate function.
    """

    def __init__(
        self,
        env: Env | str,
        states,
        style=None,
        size: int | None = None,
        variation: dict | None = None,
        frames: bool = True,
        device: str = "cuda",
        **env_kw,
    ):
        self.frames = bool(frames)
        # The env FIRST: it is the only thing that knows ``state_dim``, and without that this
        # constructor used to accept any array at all — ``reshape(-1, st.shape[-1])`` takes its width
        # from the input rather than from the env, so a frame stack became a corpus of states here and
        # only failed (or did not fail) much later, at render time.
        self._init_env(env, size, variation, device, env_kw)
        st = check_states(self.env, states, "StateDataset", leading=True, move=False)
        self.states = _states_tensor(st).reshape(-1, self.env.state_dim)
        self.style = None
        if style is not None:
            self.style = check_style(self.env, style, self.states.shape[0], "StateDataset(style=)")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def _probe_state(self) -> torch.Tensor:
        return self.states[:1].to(self.device)

    def __getitem__(self, i):
        """One sample for an integer index, the assembled batch for a sequence of them.

        The sequence form is what a ``BatchSampler`` hands a ``DataLoader`` built with
        ``batch_size=None``, and that combination collates with torch's own default — so the fastest
        way to iterate this dataset needs no argument from this module. Measured on PushT at B=1024:
        2.886 ms/batch that way against 2.907 through ``__getitems__`` + ``collate_batched``, 5.637 for
        a per-item ``DataLoader``, and 2.774 for no loader at all.
        """
        if isinstance(i, (list, tuple, np.ndarray, torch.Tensor)):
            return self.__getitems__(i)
        return self.__getitems__([i])[0]

    def __getitems__(self, idxs):
        ix = torch.as_tensor(idxs)
        s = self.states[ix].to(self.device, non_blocking=True)
        if not self.frames:
            return s
        k = None
        if self.style is not None:
            k = (self.style if self.style.shape[0] == 1 else self.style[ix]).to(self.device)
        return self.render(s, style=k)

    @property
    def compression(self) -> dict:
        """What storing states instead of frames actually saves, for this dataset."""
        n, d = self.states.shape
        h, w, c = self.frame_shape()
        state_b = n * d * 4
        style_b = 0 if self.style is None else int(self.style.numel()) * 4
        frame_b = n * h * w * c
        return {
            "n": n,
            "state_bytes": state_b,
            "style_bytes": style_b,
            "stored_bytes": state_b + style_b,
            "frame_bytes": frame_b,
            "frame_shape": (h, w, c),
            "ratio": frame_b / max(state_b + style_b, 1),
        }


class TrajectoryDataset(_RenderingDataset):
    """``(n_traj, T, state_dim)`` + per-trajectory style, yielding windows that respect episode bounds.

    The unit a world model trains on is a short consecutive window, not an independent frame, so the
    dataset that replaces an image corpus has to index windows. Two properties are load-bearing:

      * **A window never straddles a trajectory boundary.** The clip index is built per trajectory, and
        the constructor re-derives the last row each clip touches and asserts it is inside that
        trajectory — checked once, in O(number of clips), rather than asserted in a docstring. A window
        that crosses a boundary teaches the model a transition that never happened, and it is invisible
        in a loss curve.
      * **Style travels with the trajectory.** Appearance is part of the state (``core/style.py``), so a
        row is ``(states, style)`` and re-rendering reproduces the original frames exactly rather than
        approximately.

    Storage is flat — ``states (N, D)`` plus ``offsets``/``lengths`` — which is what makes ragged
    episodes a normal case rather than a padded special case, and mirrors
    ``stable_worldmodel.data.Dataset``. Dense ``(n_traj, T, D)`` input is accepted and flattened.

    Indexing returns states by default and frames on request (``frames=True``), because the states are
    what is stored and the frames are what is derived; a dataset that always renders would pay for
    pixels a state-space model never looks at.

    ``ds[i]`` and ``ds[[i,j,...]]`` both return a dict with a time axis that is always present, even for
    ``window=1``. A shape that depends on a parameter's value is where a downstream ``reshape`` silently
    transposes time into batch.
    """

    def __init__(
        self,
        env: Env | str,
        states,
        style=None,
        lengths=None,
        window: int = 1,
        stride: int = 1,
        frameskip: int = 1,
        frames: bool = False,
        size: int | None = None,
        variation: dict | None = None,
        device: str = "cuda",
        **env_kw,
    ):
        self._init_env(env, size, variation, device, env_kw)
        self.window = int(window)
        self.stride = int(stride)
        self.frameskip = int(frameskip)
        self.frames = bool(frames)
        if min(self.window, self.stride, self.frameskip) < 1:
            raise ValueError("window, stride and frameskip must all be >= 1")

        # An array input is checked against the ENV's state_dim before ``_flatten`` reads a width off
        # it. ``_flatten`` refuses rank 4 and above on its own, but a ``(H, W, 3)`` frame is rank 3 and
        # would have been read as ``H`` trajectories of ``W`` 3-wide states — silently, for any env
        # whose state happens to be 3 numbers wide.
        if not isinstance(states, (list, tuple)):
            check_states(self.env, states, "TrajectoryDataset", leading=True, move=False)
        self.states, self.lengths = self._flatten(states, lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        D = int(self.states.shape[-1])
        if D != self.env.state_dim:
            raise ValueError(f"{type(self.env).__name__} expects state_dim {self.env.state_dim}, got {D}")

        self.style = None
        if style is not None:
            k = check_style(self.env, style, len(self.lengths), "TrajectoryDataset(style=)", per="trajectory")
            if k.shape[0] == 1:
                k = k.repeat(len(self.lengths), 1)
            self.style = k.contiguous()

        self._build_clips()

    # construction

    @staticmethod
    def _flatten(states, lengths):
        """Dense ``(n_traj,T,D)``, a list of ``(T_i,D)``, or flat ``(N,D)`` + ``lengths`` -> flat + lengths."""
        if isinstance(states, (list, tuple)) and len(states) and not np.isscalar(states[0]):
            arrs = [_states_tensor(s) for s in states]
            if any(a.ndim != 2 for a in arrs):
                raise ValueError("a list input must be a list of (T_i, state_dim) arrays")
            lens = np.array([a.shape[0] for a in arrs], dtype=np.int64)
            return torch.cat(arrs, dim=0).contiguous(), lens
        st = _states_tensor(states)
        if st.ndim == 3:
            n, T, D = st.shape
            if lengths is None:
                return st.reshape(n * T, D).contiguous(), np.full(n, T, dtype=np.int64)
            lens = np.asarray(lengths, dtype=np.int64)
            if lens.shape != (n,) or lens.max(initial=0) > T:
                raise ValueError(f"lengths must be ({n},) with max <= {T}, got {lens.shape}")
            # Padded input: keep only the live prefix of each row so the padding can never be indexed.
            return torch.cat([st[i, : int(L)] for i, L in enumerate(lens)], dim=0).contiguous(), lens
        if st.ndim != 2:
            raise ValueError(f"states must be (n_traj,T,D), (N,D) or a list of (T_i,D); got {tuple(st.shape)}")
        if lengths is None:
            raise ValueError(
                "flat (N, state_dim) states need `lengths` — without episode boundaries a "
                "window would silently span two trajectories"
            )
        lens = np.asarray(lengths, dtype=np.int64)
        if int(lens.sum()) != int(st.shape[0]):
            raise ValueError(f"lengths sum to {int(lens.sum())} but states hold {int(st.shape[0])} rows")
        return st.contiguous(), lens

    def _build_clips(self) -> None:
        """Clip index (trajectory, absolute start row), with the boundary property re-derived and checked."""
        span = (self.window - 1) * self.frameskip + 1
        traj, start = [], []
        for t, (off, L) in enumerate(zip(self.offsets, self.lengths)):
            if L < span:
                continue
            s = np.arange(0, int(L) - span + 1, self.stride, dtype=np.int64)
            traj.append(np.full(s.shape, t, dtype=np.int64))
            start.append(off + s)
        self.clip_traj = np.concatenate(traj) if traj else np.zeros(0, np.int64)
        self.clip_start = np.concatenate(start) if start else np.zeros(0, np.int64)
        self.dropped_trajectories = int((self.lengths < span).sum())

        # The claim is "a window never straddles a trajectory boundary". Check it instead of stating it:
        # the last row every clip touches must be inside its own trajectory. O(clips), once.
        last = self.clip_start + self.frameskip * (self.window - 1)
        end = (self.offsets + self.lengths - 1)[self.clip_traj] if len(self.clip_traj) else np.zeros(0)
        if len(last) and (last > end).any():
            bad = int(np.argmax(last > end))
            raise AssertionError(
                f"clip {bad} would read row {int(last[bad])} past the end of "
                f"trajectory {int(self.clip_traj[bad])} (last row {int(end[bad])})"
            )
        self._clip_rows_cache = None

    # indexing

    @property
    def n_trajectories(self) -> int:
        return int(len(self.lengths))

    @property
    def clip_indices(self) -> list[tuple[int, int]]:
        """``(trajectory, start-within-trajectory)`` per clip. Mirrors ``stable_worldmodel``'s name."""
        return [(int(t), int(s - self.offsets[t])) for t, s in zip(self.clip_traj, self.clip_start)]

    def __len__(self) -> int:
        return int(len(self.clip_start))

    def _probe_state(self) -> torch.Tensor:
        return self.states[:1].to(self.device)

    def _rows(self, idxs: np.ndarray) -> torch.Tensor:
        """``(B, window)`` absolute row indices. Computed, not stored: the table would be B x window x 8
        bytes for a corpus that is otherwise a few megabytes."""
        starts = torch.as_tensor(self.clip_start[idxs])
        off = torch.arange(self.window, dtype=torch.int64) * self.frameskip
        return starts[:, None] + off[None, :]

    def __getitem__(self, i):
        if isinstance(i, (list, tuple, np.ndarray, torch.Tensor)):
            return self.__getitems__(i)
        out = self.__getitems__([i])
        return {k: (v[0] if torch.is_tensor(v) else v) for k, v in out.items()}

    def __getitems__(self, idxs) -> dict:
        """``(B, window, state_dim)`` states (+ style, indices, and frames when ``frames=True``).

        One gather and, if frames were asked for, one render of ``B*window`` states in a single launch —
        the flatten/reshape around it is why the caller does not have to do it.
        """
        ix = np.asarray(idxs, dtype=np.int64)
        rows = self._rows(ix)  # (B, window)
        s = self.states[rows.reshape(-1)].reshape(len(ix), self.window, -1)
        s = s.to(self.device, non_blocking=True)
        traj = torch.as_tensor(self.clip_traj[ix])
        out = {
            "states": s,
            "traj": traj,
            "start": torch.as_tensor(self.clip_start[ix] - self.offsets[self.clip_traj[ix]]),
        }
        k = None
        if self.style is not None:
            k = self.style[traj].to(self.device, non_blocking=True)  # (B, style_dim)
            out["style"] = k
        if self.frames:
            kk = None if k is None else k.repeat_interleave(self.window, dim=0)
            out["frames"] = self.render(s.reshape(-1, s.shape[-1]), style=kk).reshape(
                len(ix), self.window, *self.frame_shape()
            )
        return out

    def episode(self, i: int) -> dict:
        """The whole trajectory ``i``, ignoring the window/stride index. For evaluation and video."""
        off, L = int(self.offsets[i]), int(self.lengths[i])
        s = self.states[off : off + L].to(self.device)
        out = {"states": s, "traj": torch.tensor(i), "start": torch.tensor(0)}
        if self.style is not None:
            out["style"] = self.style[i : i + 1].to(self.device)
        if self.frames:
            out["frames"] = self.render(s, style=out.get("style"))
        return out

    @property
    def per_row_style(self) -> torch.Tensor | None:
        """``(N, style_dim)`` — the trajectory's style repeated over its own rows.

        Style is stored once per trajectory because that is where it is constant, but every renderer
        entry point takes one row per state, and expanding it by hand is where ``repeat`` gets used
        instead of ``repeat_interleave`` and every trajectory silently gets trajectory 0's appearance.
        """
        if self.style is None:
            return None
        return self.style.repeat_interleave(torch.as_tensor(self.lengths), dim=0)

    @torch.no_grad()
    def all_frames(self, chunk: int = 8192, out=None) -> torch.Tensor:
        """Re-render the whole stored corpus, in stored order: ``(N, H, W, 3)``.

        This is the operation the repo exists to make cheap — turning a state corpus back into the image
        dataset it replaces — and it is chunked, so the frames it produces may be far larger than the
        device memory that produces them.
        """
        return render_states(
            self.env,
            self.states,
            size=self.size,
            chunk=chunk,
            variation=self.variation,
            style=self.per_row_style,
            out=out,
            device=self.device,
        )

    @property
    def compression(self) -> dict:
        """What the stored corpus costs against the frames it replaces.

        Counted over the **stored rows**, not over clips: overlapping windows re-read the same rows, so
        counting clips would multiply both sides by the overlap factor and inflate nothing but the
        appearance of scale.
        """
        n, d = self.states.shape
        h, w, c = self.frame_shape()
        state_b = int(n) * int(d) * 4
        style_b = 0 if self.style is None else int(self.style.numel()) * 4
        frame_b = int(n) * h * w * c
        return {
            "n_trajectories": self.n_trajectories,
            "n_steps": int(n),
            "state_dim": int(d),
            "state_bytes": state_b,
            "style_bytes": style_b,
            "stored_bytes": state_b + style_b,
            "frame_bytes": frame_b,
            "frame_shape": (h, w, c),
            "ratio": frame_b / max(state_b + style_b, 1),
        }

    def describe(self) -> dict:
        return {
            "env": type(self.env).__name__,
            "size": self.size,
            "n_trajectories": self.n_trajectories,
            "n_steps": int(self.states.shape[0]),
            "state_dim": int(self.states.shape[-1]),
            "style_dim": int(self.env.style_dim),
            "window": self.window,
            "stride": self.stride,
            "frameskip": self.frameskip,
            "n_clips": len(self),
            "dropped_trajectories": self.dropped_trajectories,
            "has_style": self.style is not None,
            "frames": self.frames,
        }

    def __repr__(self) -> str:
        d = self.describe()
        return (
            f"TrajectoryDataset({d['env']}, {d['n_trajectories']} traj x ~"
            f"{d['n_steps'] // max(d['n_trajectories'], 1)} steps, window={d['window']}, "
            f"stride={d['stride']}, {len(self)} clips)"
        )

    # HuggingFace

    def to_hf(self, features_dtype: str = "float32"):
        """One row per trajectory: ``{states, style, length, stable_envs}``.

        Rows are trajectories rather than steps because the episode boundary is the one piece of
        structure a flat step table loses, and losing it is what lets a window straddle two episodes
        after a reload. ``Sequence(Sequence(...))`` keeps ragged episodes ragged rather than forcing a
        padded rectangle.

        float32 is preserved end to end. Arrow stores float32; ``datasets`` hands values back as Python
        floats (C doubles) and ``np.asarray(..., float32)`` narrows them back — widening then narrowing
        is exact, so the reload is bit-identical, which is what makes the frames bit-identical.

        The metadata is written TWICE, to ``info.description`` and to a constant ``stable_envs`` column,
        and the column is the one that matters. ``DatasetInfo._INCLUDED_INFO_IN_YAML`` (datasets 4.0.0)
        is ``[config_name, download_size, dataset_size, features, splits]`` — description is not in it —
        so ``push_to_hub`` writes a card without it and ``load_dataset`` comes back with an empty
        description. Measured: after ``to_parquet``/``from_parquet``, ``info.description`` is empty.
        A corpus that has forgotten which env it belongs to is not self-describing, and the column
        survives every serialization at a cost parquet dictionary-encodes to a few hundred bytes.
        """
        ds_mod = _require_datasets()
        feat = ds_mod.Features(
            {
                "states": ds_mod.Sequence(ds_mod.Sequence(ds_mod.Value(features_dtype))),
                "style": ds_mod.Sequence(ds_mod.Value(features_dtype)),
                "length": ds_mod.Value("int32"),
                HF_META_KEY: ds_mod.Value("string"),
            }
        )
        default_style = np.asarray(self.env.default_style(), np.float32)
        meta = json.dumps(self._meta())
        rows = {"states": [], "style": [], "length": [], HF_META_KEY: []}
        for i in range(self.n_trajectories):
            off, L = int(self.offsets[i]), int(self.lengths[i])
            rows["states"].append(self.states[off : off + L].numpy())
            rows["style"].append(default_style if self.style is None else self.style[i].numpy())
            rows["length"].append(L)
            rows[HF_META_KEY].append(meta)
        out = ds_mod.Dataset.from_dict(rows, features=feat)
        out.info.description = json.dumps({HF_META_KEY: self._meta()})
        return out

    def _meta(self) -> dict:
        return {
            "env": type(self.env).__name__,
            "size": int(self.size),
            "state_dim": int(self.states.shape[-1]),
            "style_dim": int(self.env.style_dim),
            "state_names": list(self.env.state_names),
            "style_names": list(self.env.style_names),
            "has_style": self.style is not None,
            "stable_envs_version": _version(),
        }

    def save_to_disk(self, path: str | os.PathLike, **kw) -> str:
        """Arrow directory via ``datasets.Dataset.save_to_disk``. Round-trips ``(states, style)``."""
        p = str(path)
        self.to_hf().save_to_disk(p, **kw)
        return p

    def push_to_hub(self, repo_id: str, **kw):
        """Thin passthrough. The wire format is the same parquet the local round-trip exercises.

        Not exercised in this repo's own runs: pushing creates a repository under someone's account,
        which is not a side effect a library example gets to have. ``examples/hf_roundtrip.py``
        therefore asserts the round-trip over ``save_to_disk``/``load_from_disk`` and over the parquet
        serialization ``push_to_hub`` uses, which is everything except the HTTP.
        """
        return self.to_hf().push_to_hub(repo_id, **kw)

    @classmethod
    def from_hf(cls, source, env: Env | str | None = None, split: str = "train", **kw) -> TrajectoryDataset:
        """Load from a hub repo id, a ``save_to_disk`` directory, or an in-memory ``datasets.Dataset``.

        ``env`` is optional when the description carries the metadata this class writes; a corpus from
        somewhere else has to be told which renderer its states belong to, and guessing from
        ``state_dim`` would pick the wrong env the moment two envs share a width.
        """
        ds_mod = _require_datasets()
        if isinstance(source, (str, os.PathLike)) and Path(source).exists():
            hf = ds_mod.load_from_disk(str(source))
        elif isinstance(source, (str, os.PathLike)):
            hf = ds_mod.load_dataset(str(source), split=split)
        else:
            hf = source
        if hasattr(hf, "keys") and not hasattr(hf, "features"):  # a DatasetDict
            hf = hf[split]

        # The column first: it is the copy that survives push_to_hub (see ``to_hf``). The description is
        # the fallback for a corpus written before the column existed.
        meta = {}
        if HF_META_KEY in hf.column_names:
            meta = json.loads(hf[HF_META_KEY][0])
        else:
            try:
                meta = json.loads(hf.info.description or "{}").get(HF_META_KEY, {})
            except (json.JSONDecodeError, TypeError, AttributeError):
                meta = {}
        env = env or meta.get("env")
        if env is None:
            raise ValueError(
                "this dataset does not name its env (no stable-envs metadata in the description) — "
                'pass env="PushT" or a live env. The states alone do not say what they mean.'
            )
        kw.setdefault("size", meta.get("size"))

        cols = hf.column_names
        arrs = [np.asarray(r, dtype=np.float32) for r in hf["states"]]
        style = None
        if "style" in cols and meta.get("has_style", True):
            style = np.asarray(hf["style"], dtype=np.float32)
        return cls(env, arrs, style=style, **kw)

    @classmethod
    def load_from_disk(cls, path: str | os.PathLike, env=None, **kw) -> TrajectoryDataset:
        return cls.from_hf(path, env=env, **kw)


def _require_datasets():
    """Import ``datasets``, or explain what to install. A missing optional dep is not a bug report.

    Without this the failure is an ``ImportError`` traceback out of the middle of a save, which reads
    like the library is broken rather than like an extra is missing.
    """
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "the HuggingFace round-trip needs the `datasets` package, which stable-envs does not "
            "require: everything else here works on plain tensors and numpy. Install it with "
            "`pip install datasets` (or `pip install stable-envs[hf]`), or save the corpus yourself — "
            "a TrajectoryDataset is `states (N, state_dim)`, `lengths (n_traj,)` and "
            "`style (n_traj, style_dim)`, three arrays that np.savez stores fine."
        ) from e
    return datasets


def _version() -> str:
    from . import __version__

    return __version__
