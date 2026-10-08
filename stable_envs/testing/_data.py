"""Private corpus utilities for certification and research tools.

Stored rows are float32 render projections, not full simulator State. Store
appearance alongside them when reproducing frames. Dataset indexing returns
already assembled batches: use collate_batched with automatic DataLoader
batching, and keep CUDA rendering in the main process (num_workers=0).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from stable_envs.env import Env, check_states, require_contiguous_out
from stable_envs.registry import make

HF_META_KEY = "stable_envs"


def _states_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).contiguous()


def check_style(env, style, rows: int, owner: str, per: str = "state") -> torch.Tensor:
    """Validate appearance width and row count; return contiguous float32 CPU rows.

    Accept one shared row or exactly ``rows`` rows. Check the last axis before
    reshaping so a transposed input cannot be silently regrouped.
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
    """Pass through assembled tensor/dict batches; collate lists with Torch's default.

    Use this with the datasets' __getitems__ hook. When rendering on CUDA, use
    num_workers=0 so workers do not create separate GPU contexts.
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
    """Render ``(..., state_dim)`` projections in chunks, preserving leading axes.

    Style is one shared row or one row per flattened state. Input stays on its
    original device until each chunk is rendered. Supply a contiguous CPU output
    buffer to keep the full frame stack out of device memory.
    """
    # Construct the environment before validating width; use input shape only to estimate batch size.
    probe_shape = tuple(getattr(states if torch.is_tensor(states) else np.asarray(states), "shape", ()))
    n_guess = int(np.prod(probe_shape[:-1])) if len(probe_shape) >= 1 else 1
    if isinstance(env, str):
        env = make(env, num_envs=max(1, min(chunk, n_guess or 1)), size=size, device=device, **env_kw)

    # Validate in place and transfer input one chunk at a time.
    st = check_states(env, states, "render_states", leading=True, move=False)
    lead = tuple(st.shape[:-1])
    st = st.contiguous().reshape(-1, env.state_dim)
    N = st.shape[0]
    S = size or env.size

    sty = None
    if style is not None:
        sty = check_style(env, style, N, "render_states(style=)")

    probe = env._render(
        st[:1].to(device), size=S, variation=variation, style=None if sty is None else sty[:1].to(device)
    )
    H, W = int(probe.shape[1]), int(probe.shape[2])

    if out is None:
        out = torch.empty((N, H, W, 3), dtype=torch.uint8, device=device)
    # Reshaping strided output may copy, causing writes to miss the caller's buffer.
    require_contiguous_out("render_states(out=)", out)
    flat = out.reshape(N, H, W, 3)
    # NumPy also exposes .device; only Torch devices have the .type interface used here.
    on_device = torch.is_tensor(flat) and flat.device.type == torch.device(device).type

    for i in range(0, N, chunk):
        s = st[i : i + chunk].to(device, non_blocking=True)
        k = None if sty is None else (sty if sty.shape[0] == 1 else sty[i : i + chunk]).to(device)
        if on_device:
            env._render(s, size=S, variation=variation, style=k, out=flat[i : i + chunk])
        else:
            flat[i : i + chunk] = env._render(s, size=S, variation=variation, style=k).cpu()
    return flat.reshape(*lead, H, W, 3) if lead else flat


class _RenderingDataset(torch.utils.data.Dataset):
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
        """Render projections with one style per state or per contiguous window."""
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
                # Repeat each window's style consecutively; tiling would mix appearances across trajectories.
                if s.shape[0] % k.shape[0]:
                    raise ValueError(f"style rows {k.shape[0]} do not divide state rows {s.shape[0]}")
                k = k.repeat_interleave(s.shape[0] // k.shape[0], dim=0)
        img = self.env._render(s, size=self.size, variation=self.variation, style=k)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def frame_shape(self) -> tuple[int, int, int]:
        """Measure ``(H, W, 3)`` with a one-row render; outputs need not be square.

        Use the dataset's first style, or the default, to avoid broadcasting an
        unrelated per-lane environment appearance.
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
                f = self.env._render(self._probe_state(), size=self.size, variation=self.variation, style=style)
            self._frame_hw = (int(f.shape[-3]), int(f.shape[-2]))
        return (*self._frame_hw, 3)

    def _probe_state(self) -> torch.Tensor:
        raise NotImplementedError


class StateDataset(_RenderingDataset):
    """Store render projections and optional appearance; render indexed batches on demand.

    ``frames=False`` returns state rows instead. Integer indexing removes the
    batch axis; a sequence of indices keeps it. Use collate_batched when a
    DataLoader calls __getitems__, and num_workers=0 for CUDA rendering.
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
        """Report uncompressed state/style bytes versus equivalent uint8 frame bytes."""
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
    """Index windows within trajectories, with optional per-trajectory appearance.

    Accept dense ``(episodes, T, D)``, ragged arrays, or flat rows with lengths.
    Windows never cross episode boundaries and retain a time axis at window=1.
    Items contain states and indices; frames=True additionally renders them.
    Batched indexing uses the same collation contract as StateDataset.
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

        # Validate array width against the environment before flattening episode/time axes.
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

    @staticmethod
    def _flatten(states, lengths):
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
            # Discard padding so clip indexing can reach only live rows.
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

        # Check the last row of every clip against its own episode boundary.
        last = self.clip_start + self.frameskip * (self.window - 1)
        end = (self.offsets + self.lengths - 1)[self.clip_traj] if len(self.clip_traj) else np.zeros(0)
        if len(last) and (last > end).any():
            bad = int(np.argmax(last > end))
            raise AssertionError(
                f"clip {bad} would read row {int(last[bad])} past the end of "
                f"trajectory {int(self.clip_traj[bad])} (last row {int(end[bad])})"
            )
        self._clip_rows_cache = None

    @property
    def n_trajectories(self) -> int:
        return int(len(self.lengths))

    @property
    def clip_indices(self) -> list[tuple[int, int]]:
        """Return each clip's trajectory ID and start offset within that trajectory."""
        return [(int(t), int(s - self.offsets[t])) for t, s in zip(self.clip_traj, self.clip_start)]

    def __len__(self) -> int:
        return int(len(self.clip_start))

    def _probe_state(self) -> torch.Tensor:
        return self.states[:1].to(self.device)

    def _rows(self, idxs: np.ndarray) -> torch.Tensor:
        starts = torch.as_tensor(self.clip_start[idxs])
        off = torch.arange(self.window, dtype=torch.int64) * self.frameskip
        return starts[:, None] + off[None, :]

    def __getitem__(self, i):
        if isinstance(i, (list, tuple, np.ndarray, torch.Tensor)):
            return self.__getitems__(i)
        out = self.__getitems__([i])
        return {k: (v[0] if torch.is_tensor(v) else v) for k, v in out.items()}

    def __getitems__(self, idxs) -> dict:
        """Return ``(B, window, D)`` states, trajectory/start indices, and optional style/frames."""
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
        """Return a whole trajectory without applying the window index."""
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
        """Repeat each trajectory's appearance over its own rows, or return None."""
        if self.style is None:
            return None
        return self.style.repeat_interleave(torch.as_tensor(self.lengths), dim=0)

    @torch.no_grad()
    def all_frames(self, chunk: int = 8192, out=None) -> torch.Tensor:
        """Render stored rows in order, using chunks and an optional output buffer."""
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
        """Compare stored rows and styles with frame bytes, counting overlapping windows only once."""
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

    def to_hf(self, features_dtype: str = "float32"):
        """Create one HuggingFace row per trajectory with states, style, length, and metadata.

        Float32 is the default storage dtype. Metadata is duplicated into a column
        because dataset descriptions may be lost during Parquet/Hub serialization.
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
        """Save the Arrow dataset and return its directory path."""
        p = str(path)
        self.to_hf().save_to_disk(p, **kw)
        return p

    def push_to_hub(self, repo_id: str, **kw):
        """Upload the trajectory dataset through datasets.Dataset.push_to_hub."""
        return self.to_hf().push_to_hub(repo_id, **kw)

    @classmethod
    def from_hf(cls, source, env: Env | str | None = None, split: str = "train", **kw) -> TrajectoryDataset:
        """Load a Hub ID, Arrow directory, or in-memory HuggingFace dataset.

        An explicit env overrides stored metadata. Without either, loading fails
        rather than guessing the renderer from state width.
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

        # Prefer the metadata column, which survives serialization; description is a fallback.
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
    from stable_envs import __version__

    return __version__
