"""Batched environments with explicit simulator State and state-driven rendering.

``reset`` and ``get_state`` return an owned TensorDict snapshot. ``set_state``
restores that snapshot under the environment's current Variation. State includes
hidden simulator fields and preserves their declared precision.

``render(state)`` draws a snapshot without advancing simulation. Renderers use a
private float32 projection that may omit dynamics fields; compact renderer rows
cannot replace a full State checkpoint.

``step`` accepts one action or a sequence per environment. Independent slices
select which post-action States and frames to return. The final State is always
returned, and validity masks distinguish executed transitions from held endpoints.

Variation configures persistent appearance and environment parameters. The output
device does not determine where physics runs; execution depends on the family.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from . import _dynamics
from ._config import get_config as _get_config
from ._style import StyleTensor
from .spaces import Dict as VariationDict
from .specs import CompositeSpec, Spec

#: Full simulator State is a TensorDict with entity-grouped keys and batch size [num_envs].
State = TensorDict

#: Step results always contain final State; task outputs exist only where the family defines them.
StepResult = TensorDict

#: Variation accepts leaf overrides or a flat/batched compiled configuration row.
VariationLike = Mapping[str, Any] | torch.Tensor | np.ndarray

_MAX_LISTED = 8


def require_contiguous_out(owner: str, out) -> None:
    """Require a contiguous uint8 output buffer.

    Render backends write through flat views. A strided destination may silently
    write the wrong locations; callers must copy into such a view themselves.
    """
    if out is None:
        return
    if torch.is_tensor(out):
        ok = out.is_contiguous()
    else:  # NumPy reshape may copy a strided destination.
        flags = getattr(out, "flags", None)
        ok = flags is None or bool(flags["C_CONTIGUOUS"])
    # Check dtype before any backend can silently convert the destination.
    dtype = getattr(out, "dtype", None)
    if dtype is not None and dtype not in (torch.uint8, np.dtype(np.uint8)):
        raise ValueError(
            f"{owner}: `out` must be uint8 and this one is {dtype}. A frame is 3 bytes per pixel; a "
            f"float destination accepts the write without complaining and hands back 0..255 floats "
            f"where pixels were asked for."
        )
    if ok:
        return
    raise ValueError(
        f"{owner}: `out` must be contiguous, and this one is not. The renderer writes the frames "
        f"linearly through a flat view of the destination, and a view of a strided tensor keeps its "
        f"strides, so the frames would land across the neighbouring slices of your buffer instead of "
        f"in it — silently, since nothing about the write is invalid. This is what a time or camera "
        f"axis in front of the batch axis produces: `buf[:, t]` of a (B, T, H, W, 3) buffer is "
        f"strided. Either assign (`buf[:, t] = env.render(states)`), which costs one copy, or give "
        f"the buffer a layout whose frame axis is outermost (`buf[t]` of a (T, B, H, W, 3) buffer is "
        f"contiguous and can be rendered into directly)."
    )


def _guard_render_out(fn):
    """Validate output buffers and compact input rows before a subclass renders."""

    @functools.wraps(fn)
    def render(self, *args, **kw):
        out = kw.get("out", args[5] if len(args) > 5 else None)
        if out is not None:
            require_contiguous_out(f"{type(self).__name__}.render", out)
        if args:
            args = (render_states_arg(self, args[0]), *args[1:])
        else:
            kw["states"] = render_states_arg(self, kw.get("states"))
        return fn(self, *args, **kw)

    render._se_out_guarded = True
    return render


def _as_state_tensor(owner: str, states) -> torch.Tensor:
    """Convert array-like input without casting away dtype information used by validation."""
    if isinstance(states, Mapping):
        raise TypeError(
            f"{owner}: got a {type(states).__name__}, not an array of states. A TrajectoryDataset "
            f"batch is a dict — the states are under the 'states' key, so pass batch['states'] "
            f"(and note it carries a time axis: (B, window, state_dim))."
        )
    if torch.is_tensor(states):
        return states
    if (
        isinstance(states, (list, tuple))
        and len(states)
        and all(torch.is_tensor(x) or isinstance(x, np.ndarray) for x in states)
    ):
        parts = [torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x)) for x in states]
        # TensorDataset loaders wrap a batch in a singleton sequence; stacking it would add an axis.
        # Longer sequences represent separate samples and are stacked.
        return parts[0] if len(parts) == 1 else torch.stack(parts)
    try:
        return torch.as_tensor(states)
    except Exception as e:  # noqa: BLE001 — re-raised with the fix
        raise TypeError(
            f"{owner}: could not read {type(states).__name__} as an array of states ({e}). It wants "
            f"a (batch, state_dim) float array — a torch tensor, a numpy array, or anything "
            f"torch.as_tensor accepts."
        ) from e


def _nonfinite_report(s: torch.Tensor, names: tuple[str, ...]) -> str | None:
    """Describe nonfinite rows and named columns, or return None when all values are finite."""
    bad = ~torch.isfinite(s)
    if not bool(bad.any()):
        return None
    rows = torch.nonzero(bad.any(1), as_tuple=False).flatten().tolist()
    cols = torch.nonzero(bad.any(0), as_tuple=False).flatten().tolist()
    named = [f"{names[c]} (column {c})" if c < len(names) else f"column {c}" for c in cols[:_MAX_LISTED]]
    kinds = []
    if bool(torch.isnan(s).any()):
        kinds.append("NaN")
    if bool(torch.isinf(s).any()):
        kinds.append("inf")
    more_r = f" and {len(rows) - _MAX_LISTED} more" if len(rows) > _MAX_LISTED else ""
    more_c = f" and {len(cols) - _MAX_LISTED} more" if len(cols) > _MAX_LISTED else ""
    return (
        f"{len(rows)} of {s.shape[0]} state(s) hold {'/'.join(kinds)}: "
        f"row(s) {rows[:_MAX_LISTED]}{more_r}, in {', '.join(named)}{more_c}"
    )


def render_states_arg(env, states) -> torch.Tensor:
    """Resolve private renderer input to live rows or a validated compact projection.

    CUDA tensors skip the nonfinite scan to avoid a host synchronization. Array
    inputs are scanned before transfer; full State inputs use the layout projection.
    """
    env._check_open("render")
    if states is None:
        return env._get_render_state()
    if isinstance(states, TensorDict):
        return env._pack_render_state(states)
    scan = not (torch.is_tensor(states) and states.device.type == "cuda")
    return check_states(env, states, f"{type(env).__name__}.render", leading=True, scan=scan)


def _split_appearance(variation: VariationLike | None, style):
    """Separate mapping overrides from compiled appearance rows; reject conflicting rows."""
    if variation is not None and not isinstance(variation, Mapping):
        if style is not None:
            raise TypeError("pass appearance as variation= (dict or row), not both a row and style=")
        return None, variation
    return variation, style


def _names_phrase(names: tuple[str, ...], limit: int = 8) -> str:
    if not names:
        return ""
    shown = ", ".join(names[:limit]) + (", ..." if len(names) > limit else "")
    return f" ({shown})"


def check_states(
    env, states, owner: str | None = None, leading: bool = False, scan: bool = True, move: bool = True
) -> torch.Tensor:
    """Validate compact renderer rows and convert them to float32.

    Accept ``(B, state_dim)`` or flat input containing a whole number of rows;
    ``leading=True`` also permits extra leading axes. Reject image-shaped input
    before casting and, when ``scan=True``, nonfinite values. Preserve input shape.

    ``move=False`` preserves the input device for callers that stream chunks.
    This helper does not impose an environment batch-size limit.
    """
    owner = owner or f"{type(env).__name__}.set_state"
    D = int(env.state_dim)
    names = tuple(getattr(env, "state_names", ()) or ())
    raw = _as_state_tensor(owner, states)
    shape = tuple(raw.shape)
    dt = str(raw.dtype).replace("torch.", "")

    if raw.ndim == 0:
        raise ValueError(
            f"{owner}: got a scalar. A state is {D} number(s)"
            f"{_names_phrase(names)}, and a batch of them is (batch, {D})."
        )

    # Identify images before rank validation, including uint8 images whose width matches state_dim.
    if raw.ndim >= 3 and (raw.dtype == torch.uint8 or (shape[-1] in (1, 3, 4) and shape[-1] != D)):
        cast = (
            " If these really are states stored as uint8, say so: states.astype('float32')."
            if raw.dtype == torch.uint8
            else ""
        )
        raise ValueError(
            f"{owner}: expected states shaped (batch, {D}); got apparent FRAMES "
            f"with shape {shape} and dtype {dt}. Supply state coordinates"
            f"{_names_phrase(names, 6)}, not image pixels.{cast}"
        )

    if raw.ndim > 2 and not leading:
        raise ValueError(
            f"{owner}: states are 2-D (batch, {D}); got {raw.ndim} axes: {shape}. "
            f"For a state corpus with time or trajectory axes, explicitly reshape to (-1, {D}) "
            "and construct an environment with that batch size. Image pixels are not State."
        )

    if raw.ndim >= 2 and shape[-1] != D:
        raise ValueError(
            f"{owner}: {type(env).__name__} expects state_dim {D}, got {shape[-1]} "
            f"(shape {shape}, dtype {dt}). Each state has {D} coordinates"
            f"{_names_phrase(names, 12)}. Check env.state_names / env.describe()."
        )

    if raw.ndim == 1 and raw.numel() % D:
        raise ValueError(
            f"{owner}: a flat input must be a whole number of {D}-wide states, and {raw.numel()} is "
            f"not a multiple of {D} (input shape {shape}, dtype {dt}). One state is {D} number(s)"
            f"{_names_phrase(names, 12)}; a batch is (batch, {D})."
        )

    if raw.numel() == 0:
        raise ValueError(
            f"{owner}: no states to render — got 0 of them (input shape {shape}). An empty batch "
            f"produces an empty frame buffer, which is almost never what a training step wants; "
            f"check the sampler, or the slice that produced it."
        )

    if scan:
        bad = _nonfinite_report(raw.reshape(-1, D), names)
        if bad:
            raise ValueError(
                f"{owner}: {bad}. A non-finite state renders SILENTLY — the rasterizer clamps it and "
                f"draws somewhere plausible, or draws nothing — so the frame looks like data and the "
                f"model trains on it. Check the step that produced these states (a diverged rollout "
                f"and a divide-by-zero in a normalisation both land here)."
            )
    raw = raw.detach() if torch.is_tensor(raw) else raw
    return raw.to(device=env.device, dtype=torch.float32) if move else raw.to(dtype=torch.float32)


#: Fidelity declarations select exact-pixel or measured visual-agreement checks.
#: A declared tier alone is not evidence that those checks passed.
FIDELITY_TIERS = ("parity", "visual")

#: Distinguish a renderer from a fixed learned approximation of its output.
RENDER_DECODERS = ("renderer", "learned")

#: Visual declarations report both measured metrics and the acceptance limits to assert.
REQUIRED_VISUAL_FIDELITY = (
    "mean_abs",
    "within8_frac",
    "ssim",
    "mean_abs_max",
    "within8_min",
    "ssim_min",
    "measured_at",
)

#: Source dependency versions bind reference evidence; mismatches block its acceptance.
#: Candidate implementation versions belong in measured_with instead.
REFERENCE_VERSIONS_KEY = "reference_versions"

#: Candidate package versions record measurement provenance without pinning acceptance to them.
MEASURED_WITH_KEY = "measured_with"


def reference_versions_of(cls) -> dict[str, str]:
    """Return source-version declarations from either fidelity tier, or an empty dict."""
    out = dict(getattr(cls, REFERENCE_VERSIONS_KEY, {}) or {})
    out.update((getattr(cls, "visual_fidelity", {}) or {}).get(REFERENCE_VERSIONS_KEY) or {})
    return out


def _check_reference_versions_shape(cls, where: str, rv) -> None:
    if rv is None:
        return
    if not isinstance(rv, dict) or not rv:
        raise ValueError(
            f"{cls.__name__}: {where} must be a non-empty dict of package -> exact version string, "
            f"got {rv!r}. An empty or non-dict value declares nothing; leave the field out instead."
        )
    for k, v in rv.items():
        if not (isinstance(k, str) and k and isinstance(v, str) and v.strip()):
            raise ValueError(
                f"{cls.__name__}: {where}[{k!r}] = {v!r}; every entry is a package name mapped to the "
                f"exact version string the declaration was measured against (e.g. 'mujoco': "
                f"'3.11.0'), so the certifier can compare it with what the measuring interpreter "
                f"recorded."
            )


class Env:
    """Base class for batched simulation, restorable State, and rendering."""

    name: str = ""
    family: str = ""
    domain: str = ""
    #: Width and labels of the private compact renderer projection.
    state_dim: int = 0
    state_names: tuple[str, ...] = ()
    requires_cuda: bool = False
    default_size: int = 96
    reference_id: str | None = None
    #: Variation leaves that change dynamics and must be excluded from appearance-only checks.
    dynamics_variation_leaves: frozenset[str] = frozenset()
    #: Minimum exact interior-pixel fraction; lowering it requires a mechanism in parity_note.
    parity_interior_min: float | None = 1.0
    #: Override the shared all-pixel exactness threshold, or use its default with None.
    parity_all_exact_min: float | None = None
    #: Explain any relaxed exact-pixel threshold and the remaining mismatch.
    parity_note: str = ""
    #: Select the parity or visual acceptance checks.
    fidelity_tier: str = "parity"
    #: Measured agreement and acceptance thresholds; required only for the visual tier.
    visual_fidelity: dict[str, Any] = {}
    #: Explain residual visual differences and limits on using the frames.
    visual_note: str = ""
    #: Declare whether frames come from a renderer or a learned approximation.
    render_decoder: str = "renderer"
    #: Agreement under sampled Variation; required for learned decoders.
    variation_fidelity: dict[str, Any] = {}
    #: Source package versions for parity evidence; visual evidence stores these in visual_fidelity.
    reference_versions: dict[str, str] = {}

    #: Entries are (key_path, width, render_offset[, public_dtype]); None offsets denote hidden fields.
    #: Hidden fields must be supplied by the simulator, not reconstructed from compact rows.
    state_layout: tuple[tuple, ...] = ()

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str | None = None,
        backend: str = "auto",
        seed: int = 0,
    ):
        """Create ``num_envs`` lanes; ``device="auto"`` selects CUDA when available, otherwise CPU."""
        device = self.resolve_device(device)
        if self.requires_cuda and not str(device).startswith("cuda"):
            raise ValueError(
                f"{type(self).__name__} requires CUDA; got device={device!r}. "
                "This environment has no CPU backend and will not fall back silently."
            )
        self.B = int(num_envs)
        self.size = int(size or self.default_size)
        self.ss = int(supersample)
        self.device = torch.device(device)
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.variation_space: VariationDict = self.make_variation_space()
        self._render_state = torch.zeros(self.B, self.state_dim, device=self.device, dtype=torch.float32)
        # Track valid rows separately while keeping renderer buffer addresses stable.
        self._live_rows = self.B
        self._style = StyleTensor.from_space(self.variation_space, self.B, device=self.device)
        # Explicit close controls resource lifetime; no finalizer is required.
        self._closed = False
        self._seed = int(seed)

    @staticmethod
    def resolve_device(device: str | None) -> str:
        if device is None:
            device = _get_config()["default_device"] or "auto"
        device = str(device)
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def class_name(self) -> str:
        return type(self).__name__

    @property
    def id(self) -> str:
        """Stable environment slug, distinct from its Python class name."""
        return self.name or type(self).__name__

    @property
    def render_state_dim(self) -> int:
        """Width of the private float32 renderer projection, which may omit simulator fields."""
        return int(self.state_dim)

    @property
    def render_state_names(self) -> tuple[str, ...]:
        return tuple(self.state_names)

    def __init_subclass__(cls, **kw):
        """Wrap each new ``_render`` override with shared input and output-buffer validation."""
        super().__init_subclass__(**kw)
        fn = cls.__dict__.get("_render")
        if fn is not None and not getattr(fn, "_se_out_guarded", False):
            cls._render = _guard_render_out(fn)

    def make_variation_space(self) -> VariationDict:
        """Declare persistent configuration leaves; an empty space exposes no variation."""
        raise NotImplementedError

    def scene(self, states: torch.Tensor, variation: VariationLike | None = None, style=None) -> Scene2D:
        """Build a Scene2D from ``(B, render_state_dim)`` compact rows.

        Primitive renderers implement this hook; other backends override ``_render``.
        Families whose field of view depends on output resolution also accept ``size``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not build a Scene2D; it must override render() instead")

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample ``(batch, state_dim)`` compact renderer rows using the family sampler."""
        raise NotImplementedError

    def _require_state(self, operation: str) -> None:
        self._check_open(operation)
        if not self.has_dynamics:
            _dynamics.no_dynamics(self, operation)

    def _validate_state(self, state: State, *, batch: int | None = None) -> None:
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        for key, width, _ in self._layout():
            if key not in state.keys(include_nested=True, leaves_only=True):
                raise ValueError(f"Full State is missing {'.'.join(key)}; a render projection is not restorable")
            value = state[key]
            if value.shape != (*state.batch_size, width):
                raise ValueError(f"State field {'.'.join(key)} has invalid shape {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"State field {'.'.join(key)} contains non-finite values")

    def set_state(self, state: State) -> None:
        """Restore a full State with exactly ``num_envs`` lanes under the current Variation."""
        self._require_state("set_state")
        self._validate_state(state, batch=self.B)
        self._restore_state(state)
        self._live_rows = self.B

    def _restore_state(self, state: State) -> None:
        if any(offset is None for _, _, offset in self._layout()):
            raise NotImplementedError(f"{type(self).__name__} must restore its hidden simulator fields")
        self._set_render_state(self._pack_render_state(state))

    @property
    def state(self) -> State:
        """Return an independent full snapshot, as with :meth:`get_state`."""
        return self.get_state()

    def render(self, state: State | None = None, **kwargs) -> torch.Tensor:
        """Render full State, or the live snapshot when omitted, without advancing simulation.

        Return uint8 RGB frames on ``device``. Renderer options are forwarded to
        ``_render``; the supplied State may have a different batch size from this env.
        """
        state = self.get_state() if state is None else state
        self._validate_state(state)
        return self._render(self._pack_render_state(state), **kwargs)

    def _get_render_state(self) -> torch.Tensor:
        """Return a view of live compact rows, excluding unused buffer capacity."""
        return self._render_state if self._live_rows == self.B else self._render_state[: self._live_rows]

    def _set_render_state(self, states, variation: VariationLike | None = None, style=None) -> Env:
        """Load compact rows for renderer tests; this does not restore full simulator State."""
        s = check_states(self, states, f"{type(self).__name__}.set_state").reshape(-1, self.state_dim)
        n = int(s.shape[0])
        if n > self.B:
            raise ValueError(
                f"{type(self).__name__}.set_state: this env holds {self.B} environment(s) and got {n} "
                f"states. A batch may be SHORTER than the env — the short final batch of a DataLoader "
                f"with drop_last=False renders fine, and only those rows are drawn — but it cannot be "
                f"longer, because there is nowhere to put the extra {n - self.B}. Either build the env "
                f"at the batch you feed it, se.make({type(self).__name__!r}, "
                f"num_envs={n}), or render without touching the env's buffer: env._render(states), which "
                f"takes any batch."
            )
        self._render_state[:n].copy_(s)
        self._live_rows = n
        var_dict, style_row = _split_appearance(variation, style)
        if style_row is not None:
            self.set_variation(style_row)
        elif var_dict:
            self.set_variation(var_dict)
        return self

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: VariationLike | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """Render ``(..., render_state_dim)`` rows to ``(..., H, W, 3)`` uint8 frames.

        ``variation`` accepts leaf overrides or a flat/batched compiled row. ``style``
        is the internal name for the compiled appearance; public callers use Variation.
        """
        require_contiguous_out(f"{type(self).__name__}.render", out)
        # The base implementation validates directly; subclass overrides use the wrapper.
        s = render_states_arg(self, states)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        variation, style = _split_appearance(variation, style)
        sc = self.scene(s, variation, style)
        # Load the optional backend only when primitive rendering is requested.
        from .render.raster2d import render as render_scene

        img = render_scene(
            sc, size or self.size, supersample if supersample is not None else self.ss, backend=self.backend, out=out
        )
        return img.view(*lead, *img.shape[1:]) if lead else img

    @property
    def variation_dim(self) -> int:
        """Number of columns in a compiled Variation row."""
        return self.variation_space.flat_dim

    @property
    def variation_names(self) -> list[str]:
        return [p for p, _, _ in self.variation_space.flat_layout()]

    def default_variation(self) -> np.ndarray:
        """Return the declared default configuration as a ``(variation_dim,)`` NumPy row."""
        return self.variation_space.pack()

    def get_variation(self) -> torch.Tensor:
        return self._style.values

    def set_variation(self, variation: VariationLike | None = None, **overrides) -> torch.Tensor:
        """Install a configuration row or mapping of overrides and return the compiled tensor.

        Keyword overrides take precedence over mapping entries. Per-lane configuration
        is built for the number of live renderer rows.
        """
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        self._style = StyleTensor.from_space(
            self.variation_space, self._live_rows, device=self.device, values=variation, overrides=overrides or None
        )
        return self._style.values

    def sample_variation(
        self, batch: int | None = None, paths=None, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample ``(batch, variation_dim)`` independent configurations as float32 NumPy rows.

        ``paths`` restricts sampling to named leaves; other leaves retain their defaults.
        Unknown paths raise KeyError. The default batch size is ``num_envs``.
        """
        rng = rng or self.rng
        B = int(batch or self.B)
        if paths is not None:
            known = [p for p, _, _ in self.variation_space.flat_layout()]
            unknown = [p for p in paths if p not in known]
            if unknown:
                raise KeyError(
                    f"{type(self).__name__}.sample_variation: unknown variation path(s) {unknown}. "
                    f"Known paths: {known}"
                )
        out = np.tile(self.default_variation(), (B, 1))
        for path, off, w in self.variation_space.flat_layout():
            if paths is not None and path not in paths:
                continue
            node = self.variation_space
            for part in path.split("."):
                node = node[part]
            if hasattr(node, "low"):
                lo = np.broadcast_to(node.low, node.shape).astype(np.float64).reshape(-1)
                hi = np.broadcast_to(node.high, node.shape).astype(np.float64).reshape(-1)
                out[:, off : off + w] = rng.uniform(lo, hi, size=(B, w))
            elif hasattr(node, "n"):
                out[:, off : off + w] = rng.integers(0, int(node.n), size=(B, w))
        return out.astype(np.float32)

    @property
    def style_dim(self) -> int:
        """Internal appearance width; public callers use :attr:`variation_dim`."""
        return self.variation_dim

    @property
    def style_names(self) -> list[str]:
        """Internal appearance paths; public callers use :attr:`variation_names`."""
        return self.variation_names

    def default_style(self) -> np.ndarray:
        """Internal alias for default appearance; prefer :meth:`default_variation`."""
        return self.default_variation()

    def get_style(self) -> torch.Tensor:
        """Internal appearance tensor; prefer :meth:`get_variation`."""
        return self.get_variation()

    def set_style(self, style=None, **overrides) -> torch.Tensor:
        """Internal appearance setter; prefer :meth:`set_variation`."""
        return self.set_variation(style, **overrides)

    def sample_style(self, batch: int | None = None, paths=None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Internal appearance sampler; prefer :meth:`sample_variation`."""
        return self.sample_variation(batch=batch, paths=paths, rng=rng)

    def resolve_style(
        self, style=None, variation: VariationLike | None = None, batch: int | None = None
    ) -> StyleTensor:
        """Compile appearance for a renderer batch.

        Shared appearance broadcasts; per-lane appearance must match the batch size.
        Mapping overrides apply to the stored appearance without collapsing its rows.
        """
        B = int(self.B if batch is None else batch)
        variation, style = _split_appearance(variation, style)
        if style is None:
            current = self._style.for_batch(B, owner=f"{type(self).__name__}.render")
            if not variation:
                return current
            style = current.values
        return StyleTensor.from_space(self.variation_space, B, device=self.device, values=style, overrides=variation)

    def variation_values(self) -> dict:
        return self.variation_space.value

    def describe(self) -> dict[str, Any]:
        """Return environment specs, renderer fidelity declarations, and their stated limitations."""
        d = {
            "class_name": type(self).__name__,
            "id": self.id,
            "family": self.family,
            "domain": self.domain,
            "render_state_dim": int(self.render_state_dim),
            "render_state_names": list(self.render_state_names),
            "size": self.size,
            "variation_dim": int(self.variation_dim),
            "variation_names": list(self.variation_names),
            "reference_id": self.reference_id,
            "fidelity_tier": self.fidelity_tier,
            "render_decoder": self.render_decoder,
            "variations": self.variation_space.names(),
        }
        if self.fidelity_tier == "visual":
            d["visual_fidelity"] = dict(self.visual_fidelity)
            d["visual_note"] = self.visual_note
            d["not_for"] = (
                "evaluating a policy on official pixels after training on these — the "
                "frames are a different render of the same scene, not the same frame"
            )
        if self.variation_fidelity:
            d["variation_fidelity"] = dict(self.variation_fidelity)
        rv = reference_versions_of(type(self))
        if rv:
            d[REFERENCE_VERSIONS_KEY] = rv
        if self.render_decoder == "learned" and "not_for" in d:
            d["not_for"] += (
                ", and here that different render is a LEARNED approximation: a fixed "
                "neural checkpoint drew part of it, so detail on textured surfaces is "
                "synthesised rather than measured"
            )
        else:
            d["parity_interior_min"] = self.parity_interior_min
        return d

    @property
    def num_envs(self) -> int:
        """Number of simulation lanes owned by this environment."""
        return self.B

    def __len__(self) -> int:
        return self.B

    @property
    def closed(self) -> bool:
        return self._closed

    def _layout(self) -> tuple[tuple[tuple[str, ...], int, int | None], ...]:
        """Resolve ``(key_path, width, render_offset)`` entries for this instance.

        An empty declaration uses a single State field spanning the compact row.
        An offset of None denotes a simulator field absent from the renderer projection.
        """
        if self.state_layout:
            return tuple(
                (
                    tuple(e[0]) if isinstance(e[0], (tuple, list)) else (e[0],),
                    int(e[1]),
                    None if e[2] is None else int(e[2]),
                )
                for e in self.state_layout
            )
        return ((("state",), int(self.state_dim), 0),)

    def _layout_dtypes(self) -> dict[tuple[str, ...], torch.dtype]:
        """Return explicitly declared public dtypes by State key path."""
        out = {}
        for e in self.state_layout or ():
            if len(e) >= 4 and e[3] is not None:
                key = tuple(e[0]) if isinstance(e[0], (tuple, list)) else (e[0],)
                out[key] = e[3] if isinstance(e[3], torch.dtype) else getattr(torch, str(e[3]))
        return out

    def _check_layout(self) -> tuple[tuple[tuple[str, ...], int, int | None], ...]:
        lay = self._layout()
        covered = sum(w for _, w, o in lay if o is not None)
        if covered != int(self.state_dim):
            raise ValueError(
                f"{type(self).__name__}.state_layout covers {covered} of the {self.state_dim} "
                f"column(s) of the packed row. Every packed column belongs to exactly one field, "
                f"because pack/unpack must round-trip the frozen corpus layout unchanged; a field "
                f"that is dynamical state the row does not carry declares offset None instead."
            )
        return lay

    def _pack_render_state(self, state: TensorDict) -> torch.Tensor:
        """Project State to float32 renderer rows, omitting fields without render offsets."""
        if not isinstance(state, TensorDict):
            return check_states(self, state, f"{type(self).__name__}.pack", leading=True)
        lay = self._check_layout()
        n = int(state.batch_size[0]) if state.batch_size else 1
        out = torch.zeros(n, int(self.state_dim), device=self.device, dtype=torch.float32)
        for key, w, off in lay:
            if off is None:
                continue
            try:
                v = state[key]
            except KeyError as e:
                raise KeyError(
                    f"{type(self).__name__}.pack: the state is missing {'.'.join(key)}. This env's "
                    f"state keys are {['.'.join(k) for k, _, _ in lay]}."
                ) from e
            out[:, off : off + w] = torch.as_tensor(v, device=self.device, dtype=torch.float32).reshape(n, w)
        return out

    def _unpack_render_state(self, rows) -> TensorDict:
        """Copy fields represented in compact rows into a TensorDict.

        Hidden simulator fields remain absent; this cannot reconstruct full State.
        """
        lay = self._check_layout()
        dtypes = self._layout_dtypes()
        r = check_states(self, rows, f"{type(self).__name__}.unpack").reshape(-1, int(self.state_dim))
        n = int(r.shape[0])
        out: dict = {}
        for key, w, off in lay:
            if off is None:
                continue
            dt = dtypes.get(key, torch.float32)
            # Force a copy even at the same dtype so later buffer writes cannot mutate the snapshot.
            v = r[:, off : off + w].to(dt, copy=True)
            node = out
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = v
        return TensorDict(out, batch_size=[n], device=self.device)

    def _hidden_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Supply live simulator fields whose layout entries have no render offset."""
        return {}

    def _live_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Supply drawn fields at simulator precision before the float32 projection."""
        return {}

    def get_state(self) -> State:
        """Return an independent full simulator snapshot, including hidden fields and native precision."""
        self._require_state("get_state")
        td = self._unpack_render_state(self._get_render_state())
        hidden = {
            tuple(k) if isinstance(k, (tuple, list)) else (k,): v for k, v in (self._hidden_state() or {}).items()
        }
        n = int(td.batch_size[0])
        dtypes = self._layout_dtypes()
        live = {tuple(k) if isinstance(k, (tuple, list)) else (k,): v for k, v in (self._live_state() or {}).items()}
        for key, w, off in self._layout():
            if off is not None and key in live:
                td[key] = (
                    torch.as_tensor(live[key], device=self.device, dtype=dtypes.get(key, torch.float32))
                    .reshape(-1, w)[:n]
                    .clone()
                )
        missing = []
        for key, w, off in self._layout():
            if off is not None:
                continue
            if key not in hidden:
                missing.append(".".join(key))
                continue
            td[key] = torch.as_tensor(hidden[key], device=self.device, dtype=dtypes.get(key, torch.float32)).reshape(
                -1, w
            )[:n]
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__}.state: {missing} is declared in state_layout with offset "
                f"None — real dynamical state that the packed row does not carry — and "
                f"{type(self).__name__}._hidden_state() does not return it. Unfilled, it would be "
                f"reported as ZEROS, which is a wrong answer rather than a failure: env.state would "
                f"look fine while restoration would change subsequent step(action) results. "
                f"Return {{{missing[0]!r}: <the live tensor>}} from _hidden_state()."
            )
        return td.clone()

    def _step_result(self, state: TensorDict, **extras) -> TensorDict:
        """Build a result with State and only the task outputs supplied by the family."""
        td = {"state": state}
        for k, v in extras.items():
            if v is None:
                continue
            td[k] = torch.as_tensor(v, device=self.device).reshape(int(state.batch_size[0]))
        return StepResult(td, batch_size=[int(state.batch_size[0])], device=self.device)

    def state_names_of(self, key) -> list[str]:
        """Return renderer column names for a State field; hidden fields have no columns."""
        key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        for k, w, off in self._layout():
            if k == key:
                return (
                    []
                    if off is None
                    else [
                        self.state_names[i] if i < len(self.state_names) else f"column {i}"
                        for i in range(off, off + w)
                    ]
                )
        raise KeyError(f"{type(self).__name__}: no state key {'.'.join(key)}")

    @property
    def state_spec(self) -> CompositeSpec:
        """Describe public State fields per lane, including declared dtypes and bounds."""
        bounds = dict(getattr(self, "state_bounds", {}) or {})
        out: dict = {}
        for key, w, _ in self._layout():
            lo, hi = bounds.get(key, (None, None))
            names = tuple(self.state_names_of(key)) if len(key) and w == len(self.state_names_of(key)) else ()
            spec = Spec(
                shape=(w,),
                low=lo,
                high=hi,
                dtype=self._layout_dtypes().get(key, torch.float32),
                device=self.device,
                names=names,
            )
            node = out
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = spec
        return CompositeSpec(out, device=self.device)

    @property
    def action_spec(self) -> Spec:
        """Describe one lane's action; raise NotImplementedError when no action is declared."""
        dim = getattr(self, "action_dim", None)
        if dim is None:
            _dynamics.no_dynamics(self, "action_spec")
        return Spec(
            shape=(int(dim),),
            dtype=getattr(self, "action_dtype", torch.float32),
            low=getattr(self, "action_low", None),
            high=getattr(self, "action_high", None),
            device=self.device,
        )

    @property
    def frame_spec(self) -> Spec:
        """Describe one uint8 RGB frame, with shape ``(H, W, 3)``."""
        h, w = getattr(self, "frame_shape", (self.size, self.size))
        return Spec(shape=(int(h), int(w), 3), dtype=torch.uint8, low=0, high=255, device=self.device)

    @property
    def variation_spec(self) -> Spec:
        """Describe compiled Variation columns, expanding vector leaves into individual names."""
        lo, hi, names = [], [], []
        for path, off, w in self.variation_space.flat_layout():
            # Vector leaves need one label per compiled column.
            names.extend([path] if w == 1 else [f"{path}[{i}]" for i in range(w)])
            node = self.variation_space
            for part in path.split("."):
                node = node[part]
            if hasattr(node, "low"):
                lo.extend(np.broadcast_to(node.low, node.shape).astype(np.float64).reshape(-1))
                hi.extend(np.broadcast_to(node.high, node.shape).astype(np.float64).reshape(-1))
            elif hasattr(node, "n"):
                lo.extend([0.0] * w)
                hi.extend([float(node.n) - 1.0] * w)
            else:
                lo.extend([-np.inf] * w)
                hi.extend([np.inf] * w)
        return Spec(
            shape=(int(self.variation_dim),),
            low=np.asarray(lo, np.float64),
            high=np.asarray(hi, np.float64),
            device=self.device,
            names=tuple(names),
        )

    def reset(self, seed: int | None = None) -> TensorDict:
        """Reset the environment and return full State; an explicit seed restarts its sampler."""
        self._require_state("reset")
        self._check_open("reset")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self._seed = int(seed)
        self._set_render_state(self.sample_states(self.B, self.rng))
        return self.state

    def step(self, actions, *, states=slice(None), frames=slice(-1, None)) -> TensorDict:
        """Advance ``[B, A]`` or ``[B, T, A]`` actions and select post-action outputs.

        ``states`` and ``frames`` accept slices or None. Defaults retain every State and
        render the final endpoint. Selected outputs keep a time axis, even for one action.
        The final State is always returned under ``state``.

        Within a sequence, finished lanes hold their endpoint without resetting.
        ``valid``, ``state_valid``, and ``frame_valid`` mark executed transitions;
        task outputs at unexecuted slots are zero.
        """
        from ._stepping import step

        return step(self, actions, states=states, frames=frames)

    def _step(self, actions) -> TensorDict:
        """Advance one transition per lane without rendering or sequence dispatch."""
        _dynamics.no_dynamics(self, "step")

    @property
    def has_dynamics(self) -> bool:
        return _dynamics.has_dynamics(self)

    def _check_open(self, what: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__}.{what}: this environment is closed. "
                "Create a new environment before using its released resources."
            )

    def close(self) -> None:
        """Release instance-owned resources; repeated calls have no effect."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self, "_close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# Supply the base step method for override detection without introducing a reverse import.
Env._se_dynamics_base = Env
