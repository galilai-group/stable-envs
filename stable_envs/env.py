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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

from ._core import dynamics as _dynamics
from ._core.config import get_config as _get_config
from ._core.config import log as _log
from ._core.style import StyleTensor
from .spaces import Dict as VariationDict
from .specs import CompositeSpec, Spec

if TYPE_CHECKING:
    from .render.scene import Scene2D

#: Full simulator State is a TensorDict with entity-grouped keys and batch size [num_envs].
State = TensorDict

#: Step results always contain final State; task outputs exist only where the family defines them.
StepResult = TensorDict

#: Variation accepts leaf overrides or a flat/batched compiled configuration row.
VariationLike = Mapping[str, Any] | torch.Tensor | np.ndarray

_MAX_LISTED = 8


@functools.cache
def _log_auto_cpu() -> None:
    _log("INFO", "No CUDA device is available, so device='auto' chose the CPU. Pass device='cuda' to require a GPU.")


def require_contiguous_out(owner: str, out) -> None:
    """Raise ValueError unless ``out`` is None or a contiguous uint8 buffer.

    Render backends write through a flat view of ``out``, so a strided destination
    would receive frames at the wrong locations.
    """
    if out is None:
        return
    if torch.is_tensor(out):
        ok = out.is_contiguous()
    else:  # NumPy reshape may copy a strided destination.
        flags = getattr(out, "flags", None)
        ok = flags is None or bool(flags["C_CONTIGUOUS"])
    dtype = getattr(out, "dtype", None)
    if dtype is not None and dtype not in (torch.uint8, np.dtype(np.uint8)):
        raise ValueError(f"{owner}: `out` must be uint8, got {dtype}.")
    if ok:
        return
    raise ValueError(
        f"{owner}: `out` must be contiguous; the renderer writes through a flat view of it. "
        f"`buf[:, t]` of a (B, T, H, W, 3) buffer is strided: assign instead "
        f"(`buf[:, t] = env.render(states)`), or render into `buf[t]` of a (T, B, H, W, 3) buffer."
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
    """Convert array-like input to a tensor, keeping the dtype that validation inspects."""
    if isinstance(states, Mapping):
        raise TypeError(
            f"{owner}: got a {type(states).__name__}, not an array of states. A TrajectoryDataset "
            f"batch is a dict: pass batch['states'], which is (B, window, state_dim)."
        )
    if torch.is_tensor(states):
        return states
    if (
        isinstance(states, (list, tuple))
        and len(states)
        and all(torch.is_tensor(x) or isinstance(x, np.ndarray) for x in states)
    ):
        parts = [torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x)) for x in states]
        # TensorDataset loaders wrap a batch in a one-element list; only longer lists are stacked.
        return parts[0] if len(parts) == 1 else torch.stack(parts)
    try:
        return torch.as_tensor(states)
    except Exception as e:  # noqa: BLE001 — re-raised with the fix
        raise TypeError(
            f"{owner}: could not read {type(states).__name__} as an array of states ({e}). Pass a "
            f"(batch, state_dim) float array: a torch tensor, a numpy array, or anything "
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


def _keyed(fields) -> dict:
    return {tuple(k) if isinstance(k, (tuple, list)) else (k,): v for k, v in (fields or {}).items()}


def _as_field(value, dtype: torch.dtype) -> torch.Tensor:
    """Read a tensor as is; convert other array-likes straight to ``dtype`` so no precision is lost."""
    return value if torch.is_tensor(value) else torch.as_tensor(value, dtype=dtype)


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
            f"{owner}: no states to render (input shape {shape}); check the sampler or the slice that produced them."
        )

    if scan:
        bad = _nonfinite_report(raw.reshape(-1, D), names)
        if bad:
            raise ValueError(
                f"{owner}: {bad}. Renderers draw non-finite states without error, so they are "
                f"refused; check the step that produced these states."
            )
    raw = raw.detach() if torch.is_tensor(raw) else raw
    return raw.to(device=env.device, dtype=torch.float32) if move else raw.to(dtype=torch.float32)


#: A ``visual_fidelity`` record states how the agreement with the source was measured.
REQUIRED_VISUAL_FIDELITY = ("measured_at",)

#: Per-frame MSE against the source over pixels and channels on the 0-255 scale: the mean, the 95th
#: percentile and the worst frame of one seeded draw.
VISUAL_FIDELITY_MSE = ("mse_mean", "mse_p95", "mse_max")

#: Source dependency versions bind reference evidence; mismatches block its acceptance.
#: Candidate implementation versions belong in measured_with instead.
REFERENCE_VERSIONS_KEY = "reference_versions"

#: Candidate package versions record measurement provenance without pinning acceptance to them.
MEASURED_WITH_KEY = "measured_with"


def reference_versions_of(cls) -> dict[str, str]:
    """Return the declared source package versions, or an empty dict."""
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


@dataclass(frozen=True)
class ResetLeaf:
    """A State value ``reset`` samples, takes from ``variation_values``, or leaves at ``default``.

    ``sampled`` leaves are drawn when the options name no ``"variation"`` list.
    """

    path: str
    default: tuple[float, ...]
    sampled: bool = True
    sampleable: bool = True
    dtype: str = "float32"


def _lane_value(value, lane: int, batch: int, width: int) -> np.ndarray:
    v = np.asarray(value)
    if v.size == width:
        return v.reshape(width).copy()
    if v.size == batch * width:
        return v.reshape(batch, width)[lane].copy()
    raise ValueError(f"expected {width} value(s) shared by the batch or ({batch}, {width}) per lane, got {v.shape}")


class Env:
    """Base class for a batch of simulators with restorable State and state-driven rendering.

    Concrete environments are registered by class name and built with ``stable_envs.make``
    or their constructor. One instance owns ``num_envs`` lanes.
    """

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
    #: Variation leaves the source declares but never draws; appearance checks skip them too.
    undrawn_variation_leaves: frozenset[str] = frozenset()
    #: Measured agreement with the source renderer and its measurement setup; empty when not measured.
    visual_fidelity: dict[str, Any] = {}
    #: Where and why frames differ from the source renderer when ``visual_fidelity`` is declared.
    visual_note: str = ""
    #: Source package versions the comparison with the source used; ``visual_fidelity`` can also carry them.
    reference_versions: dict[str, str] = {}

    #: Entries are (key_path, width, render_offset[, public_dtype]); None offsets denote hidden fields.
    #: Hidden fields must be supplied by the simulator, not reconstructed from compact rows.
    state_layout: tuple[tuple, ...] = ()

    reset_leaves: tuple[ResetLeaf, ...] = ()
    #: Returns a fresh distribution tree for ``reset_leaves``; reset seeds it with ``seed + lane``.
    reset_tree_factory = None
    #: Variation leaves ``reset`` accepts in ``options["variation_values"]`` for that episode.
    reset_value_paths: frozenset[str] = frozenset()
    #: Variation leaves ``reset`` samples per lane when named in ``options["variation"]``.
    reset_sample_paths: frozenset[str] = frozenset()

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str | None = None,
        backend: str = "auto",
        seed: int = 0,
    ):
        """Create ``num_envs`` lanes.

        Args:
            num_envs: Number of lanes.
            size: Frame height and width in pixels; None uses ``default_size``.
            supersample: Samples per pixel along each axis.
            device: Output device. None uses the global ``default_device``; ``"auto"`` selects
                CUDA when available, otherwise CPU. Where physics runs depends on the family.
            backend: Renderer backend; ``"auto"`` lets the family choose.
            seed: Seed for the environment's sampler.
        """
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
        self._default_style = self._style
        self._default_physics = None
        self._variation_is_default = True
        # Explicit close controls resource lifetime; no finalizer is required.
        self._closed = False
        self._seed = int(seed)
        _log(
            "DEBUG",
            f"Built {type(self).__name__}: num_envs={self.B}, size={self.size}, device={self.device}, "
            f"backend={self.backend!r}.",
        )

    @staticmethod
    def resolve_device(device: str | None) -> str:
        """Resolve None to the global default device and ``"auto"`` to ``"cuda"`` or ``"cpu"``."""
        if device is None:
            device = _get_config()["default_device"] or "auto"
        device = str(device)
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            _log("DEBUG", f"Resolved device 'auto' to {resolved!r}.")
            if resolved == "cpu":
                _log_auto_cpu()
            return resolved
        return device

    @property
    def class_name(self) -> str:
        """Python class name, which is also the registry key."""
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
        """Column labels of the private renderer projection."""
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
        present = {(k,) if isinstance(k, str) else tuple(k) for k in state.keys(include_nested=True, leaves_only=True)}
        values = []
        for key, width, _, _ in self._state_plan():
            if key not in present:
                raise ValueError(f"Full State is missing {'.'.join(key)}; a render projection is not restorable")
            value = state[key]
            if value.shape != (*state.batch_size, width):
                raise ValueError(f"State field {'.'.join(key)} has invalid shape {tuple(value.shape)}")
            values.append((key, value))
        # One host synchronization for the whole State; the per-field pass only names the culprit.
        if values and not bool(torch.stack([torch.isfinite(v).all() for _, v in values]).all()):
            for key, value in values:
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
                f"states. A batch may be SHORTER than the env, but not longer. Build the env at this "
                f"batch, make({type(self).__name__!r}, num_envs={n}), or call env._render(states), "
                f"which takes any batch."
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
        from . import tuning
        from .render.raster2d import render as render_scene

        tuning.on_render("raster2d", size or self.size, s.shape[0], self)

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
        """Dotted leaf paths in compiled-row order; a vector leaf spans several columns."""
        return [p for p, _, _ in self.variation_space.flat_layout()]

    def default_variation(self) -> np.ndarray:
        """Return the declared default configuration as a ``(variation_dim,)`` NumPy row."""
        return self.variation_space.pack()

    def get_variation(self) -> torch.Tensor:
        """Return the installed compiled Variation as a ``(rows, variation_dim)`` float32 tensor."""
        return self._style.values

    def set_variation(self, variation: VariationLike | None = None, **overrides) -> torch.Tensor:
        """Install a configuration row or mapping of overrides and return the compiled tensor.

        The result is also the default that every later ``reset`` restores for leaves its
        options do not name. Keyword overrides take precedence over mapping entries.
        Per-lane configuration is built for the number of live renderer rows.
        """
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        style = StyleTensor.from_space(
            self.variation_space, self._live_rows, device=self.device, values=variation, overrides=overrides or None
        )
        self._install_variation(style)
        self._default_style = self._style
        self._default_physics = getattr(self, "_physics_installed", None)
        self._variation_is_default = True
        return self._style.values

    def _install_variation(self, style: StyleTensor, exact: Mapping[str, Any] | None = None) -> None:
        self._style = style
        self._configure_variation(exact)

    def _configure_variation(self, exact: Mapping[str, Any] | None = None) -> None:
        """Push the installed Variation into the simulator; ``exact`` holds float64 physics values."""
        self._configure_physics(exact)

    def _restore_variation(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Install the defaults, with ``overrides`` for this episode only."""
        if not overrides and self._variation_is_default:
            return
        style = self._default_style.for_batch(self.B, owner=f"{type(self).__name__}.reset")
        if overrides:
            style = StyleTensor.from_space(
                self.variation_space, self.B, device=self.device, values=style.values, overrides=dict(overrides)
            )
        exact = {path: overrides[path] for path in self.physics_variation_leaves if overrides and path in overrides}
        self._physics_installed = self._default_physics
        self._install_variation(style, exact or None)
        self._variation_is_default = not overrides

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
        """Return the current values of the variation space as a nested dict."""
        return self.variation_space.value

    def _reset_value_ok(self, path: str, value, lane: int) -> bool:
        """Return whether a sampled reset value is acceptable for ``lane``; rejected draws are resampled."""
        del path, value, lane
        return True

    def _reset_request(self, options: Mapping | None) -> tuple[tuple[str, ...], dict]:
        options = dict(options or {})
        default = tuple(leaf.path for leaf in self.reset_leaves if leaf.sampled)
        sampled = options.get("variation", default)
        if not isinstance(sampled, (list, tuple)):
            raise ValueError("variation option must be a Sequence containing variations names")
        values = dict(options.get("variation_values", {}))
        sampleable = {leaf.path for leaf in self.reset_leaves if leaf.sampleable} | set(self.reset_sample_paths)
        unsupported = set(sampled) - sampleable
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__}.reset does not certify sampled variation leaves {sorted(unsupported)}"
            )
        unsupported = set(values) - {leaf.path for leaf in self.reset_leaves} - set(self.reset_value_paths)
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__}.reset does not certify variation_values {sorted(unsupported)}"
            )
        return tuple(sampled), values

    def _draw_reset_lane(self, tree, lane: int, sampled, values) -> dict[str, np.ndarray]:
        out = {}
        for leaf in self.reset_leaves:
            value = np.asarray(leaf.default, leaf.dtype)
            if leaf.path in sampled:
                node = tree
                for part in leaf.path.split("."):
                    node = node[part]
                for _ in range(1000):
                    candidate = node.sample()
                    if self._reset_value_ok(leaf.path, candidate, lane):
                        value = candidate
                        break
                else:
                    raise RuntimeError(f"{leaf.path} rejection sampling did not converge")
            if leaf.path in values:
                value = _lane_value(values[leaf.path], lane, self.B, len(leaf.default)).astype(leaf.dtype)
            out[leaf.path] = value
        return out

    def _sample_variation_leaf(self, path: str, seed: int | None) -> np.ndarray:
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        rows = []
        for lane in range(self.B):
            rng = self.rng if seed is None else np.random.default_rng((int(seed), lane))
            if hasattr(node, "low"):
                rows.append(node.batch_sample(1, rng).reshape(-1))
            else:
                rows.append(np.asarray([rng.integers(0, int(node.n))]) + int(getattr(node, "start", 0)))
        return np.stack(rows)

    def draw_reset_values(self, seed: int | None = None, options: Mapping | None = None) -> dict[str, np.ndarray]:
        """Resolve SWM-style reset options into ``{path: (num_envs, ...)}`` values.

        Every ``reset`` goes through here. Variation leaves the options do not name return
        to their defaults (the registered values, or the last ``set_variation``); named
        leaves are sampled or set for this episode. Variation is installed before reset
        leaves are drawn, so their rejection tests see it.
        """
        sampled, values = self._reset_request(options)
        reset_paths = {leaf.path for leaf in self.reset_leaves}
        variation = {path: self._sample_variation_leaf(path, seed) for path in sampled if path not in reset_paths}
        for path, value in values.items():
            if path not in reset_paths:
                variation[path] = value
        self._restore_variation(variation)
        out = {path: np.asarray(value) for path, value in variation.items()}
        if not self.reset_leaves:
            return out
        lanes = []
        # Seeding replaces every generator in the tree, so one tree serves every lane.
        tree = self.reset_tree_factory()
        for lane in range(self.B):
            tree.seed(None if seed is None else int(seed) + lane)
            lanes.append(self._draw_reset_lane(tree, lane, sampled, values))
        for path in lanes[0]:
            out[path] = np.stack([row[path] for row in lanes])
        return out

    @property
    def physics_variation_leaves(self) -> tuple[str, ...]:
        layout = self._physics_layout()
        return () if layout is None else tuple(layout.leaves)

    def _physics_layout(self):
        return getattr(getattr(self, "_simulator", None), "physics_layout", None)

    def _physics_rows(self, exact: Mapping[str, Any] | None = None) -> np.ndarray:
        """Float64 rows for the current Variation.

        Style columns are float32, so a column that still encodes the installed or
        registered value keeps that float64 value; ``exact`` entries replace it.
        """
        layout = self._physics_layout()
        style = self.get_variation().detach().cpu().numpy()
        cols = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}
        installed = getattr(self, "_physics_installed", None)
        rows = np.tile(np.asarray(layout.nominal, np.float64), (self.B, 1))
        for leaf, off, width in zip(layout.leaves, layout.offsets, layout.widths, strict=True):
            o, w = cols[leaf]
            encoded = style[:, o : o + w]
            nominal = np.broadcast_to(rows[:, off : off + width], encoded.shape)
            value = np.where(encoded == nominal.astype(np.float32), nominal, encoded.astype(np.float64))
            if installed is not None:
                kept = installed[:, off : off + width]
                value = np.where(encoded == kept.astype(np.float32), kept, value)
            if exact is not None and leaf in exact:
                value = np.broadcast_to(np.asarray(exact[leaf], np.float64).reshape(-1, width), (self.B, width)).copy()
            rows[:, off : off + width] = value
        return rows

    def _configure_physics(self, exact: Mapping[str, Any] | None = None) -> None:
        if self._physics_layout() is None:
            return
        rows = self._physics_rows(exact)
        self._simulator.configure_physics(rows)
        self._physics_installed = rows

    def _adopt_physics_state(self, state: TensorDict) -> None:
        layout = self._physics_layout()
        if layout is None:
            return
        from ._core.physics import state_key

        values = self._style.values.clone()
        cols = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}
        rows = np.empty((self.B, int(sum(layout.widths))))
        for leaf, off, width in zip(layout.leaves, layout.offsets, layout.widths, strict=True):
            v = state[state_key(leaf)].detach().to(torch.float64).cpu().reshape(self.B, width)
            rows[:, off : off + width] = v.numpy()
            o, _ = cols[leaf]
            values[:, o : o + width] = v.to(device=values.device, dtype=values.dtype)
        self._style = StyleTensor(values, self.variation_space.flat_layout())
        self._physics_installed = rows
        self._variation_is_default = False

    def describe(self) -> dict[str, Any]:
        """Return environment specs and the declared comparison with the source.

        ``state_fields`` maps each State leaf, as a ``/``-joined key path, to its per-lane shape and dtype.
        """
        d = {
            "class_name": type(self).__name__,
            "id": self.id,
            "family": self.family,
            "domain": self.domain,
            "render_state_dim": int(self.render_state_dim),
            "render_state_names": list(self.render_state_names),
            "state_fields": {
                "/".join(key): {"shape": list(spec.shape), "dtype": str(spec.dtype).removeprefix("torch.")}
                for key, spec in self.state_spec.items(nested=True)
            },
            "size": self.size,
            "variation_dim": int(self.variation_dim),
            "variation_names": list(self.variation_names),
            "reference_id": self.reference_id,
            "variations": self.variation_space.names(),
        }
        if self.visual_fidelity:
            d["visual_fidelity"] = dict(self.visual_fidelity)
            d["visual_note"] = self.visual_note
        rv = reference_versions_of(type(self))
        if rv:
            d[REFERENCE_VERSIONS_KEY] = rv
        return d

    @property
    def num_envs(self) -> int:
        """Number of simulation lanes owned by this environment."""
        return self.B

    def __len__(self) -> int:
        return self.B

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
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
                f"column(s) of the packed row. Every packed column belongs to exactly one field so "
                f"pack/unpack round-trip; a State field the row does not carry declares offset None."
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
        """Supply live simulator fields whose layout entries have no render offset.

        Values may be views of live buffers; ``get_state`` copies each one exactly once.
        """
        return {}

    def _live_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Supply drawn fields at simulator precision before the float32 projection; views are fine."""
        return {}

    def _state_plan(self) -> tuple[tuple[tuple[str, ...], int, int | None, torch.dtype], ...]:
        """Return ``(key, width, render_offset, dtype)`` layout entries.

        The entries are cached while ``state_layout`` and ``state_dim`` are unchanged, unless
        a subclass computes its layout in ``_layout`` or ``_layout_dtypes``.
        """
        cls = type(self)
        static = cls._layout is Env._layout and cls._layout_dtypes is Env._layout_dtypes
        cached = self.__dict__.get("_state_plan_cache")
        if static and cached is not None and cached[0] is self.state_layout and cached[1] == self.state_dim:
            return cached[2]
        dtypes = self._layout_dtypes()
        plan = tuple((key, w, off, dtypes.get(key, torch.float32)) for key, w, off in self._layout())
        if static:
            self.__dict__["_state_plan_cache"] = (self.state_layout, self.state_dim, plan)
        return plan

    def get_state(self) -> State:
        """Return an independent full simulator snapshot, including hidden fields and native precision."""
        self._require_state("get_state")
        plan = self._state_plan()
        if sum(w for _, w, off, _ in plan if off is not None) != int(self.state_dim):
            self._check_layout()
        rows = self._get_render_state()
        if not bool(torch.isfinite(rows).all()):
            check_states(self, rows, f"{type(self).__name__}.unpack")
        n = int(rows.shape[0])
        hidden = _keyed(self._hidden_state())
        live = _keyed(self._live_state())
        tree: dict = {}
        missing = []
        # Drawn fields come first, then hidden ones, each in layout order.
        for key, w, off, dt in sorted(plan, key=lambda entry: entry[2] is None):
            if off is None:
                if key not in hidden:
                    missing.append(".".join(key))
                    continue
                value = _as_field(hidden[key], dt).reshape(-1, w)[:n]
            elif key in live:
                value = _as_field(live[key], dt).reshape(-1, w)[:n]
            else:
                value = rows[:, off : off + w]
            node = tree
            for part in key[:-1]:
                node = node.setdefault(part, {})
            # The one copy that makes the snapshot independent of every live buffer.
            node[key[-1]] = value.to(device=self.device, dtype=dt, copy=True)
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__}.state: {missing} is declared in state_layout with offset "
                f"None and {type(self).__name__}._hidden_state() does not return it. Reporting it as "
                f"ZEROS would make restored States step differently. "
                f"Return {{{missing[0]!r}: <the live tensor>}} from _hidden_state()."
            )
        return TensorDict(tree, batch_size=[n], device=self.device)

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
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self._seed = int(seed)
        self.draw_reset_values(seed)
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
        from ._core.stepping import step

        return step(self, actions, states=states, frames=frames)

    def _step(self, actions) -> TensorDict:
        """Advance one transition per lane without rendering or sequence dispatch."""
        _dynamics.no_dynamics(self, "step")

    @property
    def has_dynamics(self) -> bool:
        """Whether this environment implements ``step``; render-only environments do not."""
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
