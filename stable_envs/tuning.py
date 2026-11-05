"""Tuning knobs: chunk sizes, block sizes, worker counts and memory budgets.

A knob changes speed and memory use. Frames and State are the same for every valid value.
The library reads a knob each time it does the work. A change applies to the next call.

A lookup uses the first value it finds:

1. :func:`override` (innermost first), then :func:`set`.
2. The ``STABLE_ENVS_<KNOB>`` environment variable.
3. A value that ``stable-envs tune`` measured on this device for the render path, frame size and
   batch range.
4. A value from a hardware rule.
5. The call site's default.
6. The knob's fallback.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import os
import platform
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ._core import environ as _environ
from ._core.cache import cache_dir as _cache_dir
from ._core.config import log as _log

__all__ = ["Knob", "KNOBS", "RENDER_PATHS", "get", "set", "reset", "override", "describe", "cache_path"]

#: Batch size from which an untuned render path logs the ``stable-envs tune`` hint or autotunes.
HINT_LANES = 256
#: Cache entry of the knobs read without a render path (Triton kernel shapes).
DEVICE_ENTRY = "*"


def l3_bytes() -> int | None:
    """Return the L3 size of the first CPU, or None where it is not readable."""
    try:
        if os.sysconf("SC_LEVEL3_CACHE_SIZE") > 0:
            return int(os.sysconf("SC_LEVEL3_CACHE_SIZE"))
    except (AttributeError, ValueError, OSError):
        pass
    return None


def free_cuda_bytes(device) -> int | None:
    """Return the free memory of CUDA ``device``, or None without CUDA."""
    import torch

    return int(torch.cuda.mem_get_info(device)[0]) if torch.cuda.is_available() else None


def _pow2(x: float) -> int:
    return 1 << max(0, int(math.log2(max(1.0, x))))


def _cpu_rays(device) -> int | None:
    # The fallback was measured with a 35.75 MiB L3 (Xeon Platinum 8268).
    l3 = l3_bytes()
    return None if l3 is None else 1 << min(21, max(18, round(math.log2((1 << 21) * l3 / (36608 << 10)))))


def _cuda_rays(device) -> int | None:
    free = free_cuda_bytes(device)
    return None if free is None else min(1 << 24, _pow2(free / 512))


def _nvd_bytes(device) -> int | None:
    free = free_cuda_bytes(device)
    return None if free is None else min(6 << 30, free // 2)


_WARPS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Knob:
    """One knob: its fallback, the render paths that read it, and how ``tune`` sweeps it.

    ``kind`` is ``"n"`` (at least 1), ``"n0"`` (at least 0), ``"pow2"`` or ``"warps"``. ``unit`` tells
    ``tune`` which values give the same work split: ``"rows"``, ``"rays"``, ``"cpus"``, ``"bytes"``,
    ``""`` (none), or None for a knob ``tune`` does not sweep. ``step`` knobs are read when an
    environment is built and are timed by stepping. Call sites of a ``per_path=False`` knob pass no
    render path, so its measured value applies to the whole device. ``device`` is ``"cpu"``, ``"cuda"`` or ``"any"``.
    """

    name: str
    fallback: int | None
    paths: tuple[str, ...]
    device: str
    kind: str
    unit: str | None
    doc: str
    derive: Callable[[str], int | None] | None = None
    step: bool = False
    per_path: bool = True

    @property
    def env_var(self) -> str:
        """The environment variable that sets this knob."""
        return f"STABLE_ENVS_{self.name.upper()}"

    def check(self, value: Any) -> int | None:
        """Return ``value`` as a valid int, or None for unset; raise ValueError otherwise."""
        if value is None:
            return None
        ok = isinstance(value, int) and not isinstance(value, bool) and value >= (self.kind != "n0")
        if ok and self.kind == "pow2":
            ok = value & (value - 1) == 0
        if ok and self.kind == "warps":
            ok = value in _WARPS
        if not ok:
            raise ValueError(f"tuning knob {self.name!r} takes {_KINDS[self.kind]}, got {value!r}")
        return value


_KINDS = {"n": "an int >= 1", "n0": "an int >= 0", "pow2": "a power of two", "warps": f"one of {list(_WARPS)}"}
_SIM = (
    "classic-sim",
    "fetch-sim",
    "miniworld-sim",
    "ogb-sim",
    "pusht-sim",
    "pusht-render",
    "rocket-sim",
    "roadster-sim",
)
_RENDER = ("classic-render", "fetch-opengl", "miniworld-render", "roadster-render")

KNOBS: dict[str, Knob] = {
    k.name: k
    for k in (
        Knob("workers", 8, _SIM, "cpu", "n0", "cpus", "Most worker processes of a CPU simulator batch or PushT render on Linux; 0 runs in-process.", step=True),
        Knob("worker_min_lanes", 64, _SIM, "cpu", "n", None, "Smallest batch that uses worker processes or Fetch threads."),
        Knob("worker_timeout", 120, _SIM, "cpu", "n", None, "Seconds a CPU simulator worker may take to answer one request before the batch fails; starting takes up to the larger of this and 300.", per_path=False),
        Knob("render_workers", 8, _RENDER, "cpu", "n0", "cpus", "Most worker processes of a CPU render batch on Linux; 0 or 1 draws in-process."),
        Knob("render_worker_min_rows", 16, _RENDER, "cpu", "n", None, "Smallest CPU render batch that uses render worker processes."),
        Knob("fetch_threads", 4, ("fetch-sim",), "cpu", "n", "cpus", "Most simulator threads of an in-process Fetch batch.", step=True),
        Knob("dmc_threads", 8, ("dmc-sim",), "cpu", "n", "cpus", "Most mujoco.batch threads of a DM Control batch.", step=True),
        Knob("cpu3d_chunk", 16, ("dmc-trackcom",), "cpu", "n", "rows", "Frames per cpu3d ray-casting chunk."),
        Knob("dmc_cpu_ray_budget", 1 << 21, ("dmc-suite",), "cpu", "pow2", "rays", "Rays per CPU render chunk of fixed-camera DMC cells.", _cpu_rays),
        Knob("dmc_ray_budget", 1 << 24, ("dmc-suite",), "cuda", "pow2", "rays", "Rays per CUDA render chunk of fixed-camera DMC cells.", _cuda_rays),
        Knob("dmc_launch_rays", 256 * 224 * 224, ("dmc-trackcom",), "cuda", "n", "rays", "Rays per MJWarp launch of DMC TrackCom cells."),
        Knob("dmc_min_worlds", 64, ("dmc-trackcom",), "cuda", "pow2", "rows", "Fewest worlds per MJWarp launch of DMC TrackCom cells."),
        Knob("fetch_mjw_worlds", 256, ("fetch-mjwarp",), "cuda", "pow2", "rows", "Most worlds per MJWarp launch of Fetch cells."),
        Knob("ogb_mjw_worlds", 256, ("ogb-manipulation",), "cuda", "pow2", "rows", "Most worlds per MJWarp launch of OGBench Cube and Scene.", per_path=False),
        Knob("ogb_gl_workers", 8, ("ogb-manipulation",), "any", "n0", None, "Most OpenGL worker processes of an OGBench Cube and Scene batch on Linux; 0 draws in-process.", per_path=False),
        Knob("ogb_context_cache_bytes", None, ("ogb-manipulation",), "cuda", "n", None, "GPU bytes the OGBench context cache keeps per device."),
        Knob("nvd_budget_bytes", 6 << 30, ("rocket-nvdiff",), "cuda", "n", "bytes", "GPU bytes per nvdiffrast chunk of Rocket cells.", _nvd_bytes),
        Knob("nav_render_chunk", 256, ("navigation",), "any", "n", "rows", "Frames per render chunk of navigation cells."),
        Knob("navdot_block", 1024, ("navigation",), "cuda", "pow2", "", "Pixels per Triton tile of the navigation dot compositor.", per_path=False),
        Knob("navdot_warps", 4, ("navigation",), "cuda", "warps", "", "Triton warps per tile of the navigation dot compositor.", per_path=False),
        Knob("raster2d_block", 128, ("raster2d",), "cuda", "pow2", "", "Pixels per Triton program of the unculled 2-D rasterizer.", per_path=False),
        Knob("raster2d_cull_warps", 4, ("raster2d",), "cuda", "warps", "", "Triton warps per tile of the culling 2-D rasterizer.", per_path=False),
        Knob("gfxdraw_warps", 1, ("gfxdraw",), "cuda", "warps", "", "Triton warps per tile of the gfxdraw rasterizer.", per_path=False),
        Knob("raycast_block", 256, ("dmc-suite", "ogb-maze"), "cuda", "pow2", "", "Rays per Triton program of the analytic ray caster.", per_path=False),
    )
}  # fmt: skip

#: Render and simulation paths per environment class or family; the most derived class wins.
RENDER_PATHS: dict[str, tuple[str, ...]] = {
    "DMControlSuite": ("dmc-suite", "dmc-render", "dmc-sim"),
    "DMControlTrackCom": ("dmc-trackcom", "dmc-render", "dmc-sim"),
    "OGBManipBase": ("ogb-manipulation", "ogb-sim"),
    "ogbench": ("ogb-maze", "ogb-sim"),
    "fetch": ("fetch-mjwarp", "fetch-opengl", "fetch-sim"),
    "PyFlytRocketLanding": ("rocket-nvdiff", "rocket-sim"),
    "navigation": ("navigation", "raster2d"),
    "pusht": ("raster2d", "pusht-sim", "pusht-render"),
    "classic_control": ("gfxdraw", "classic-render", "classic-sim"),
    "miniworld": ("miniworld-render", "miniworld-sim"),
    "roadster": ("roadster-render", "roadster-sim"),
}

for _k in KNOBS.values():
    _environ.register(
        _k.env_var, _k.kind, None, f"Tuning knob `{_k.name}`.", lambda t, k=_k: k.check(_environ.parse_int(t))
    )
_environ.register(
    "STABLE_ENVS_AUTOTUNE", "flag", False, "1 tunes a large untuned render path on first use.", _environ.parse_flag
)
_environ.register(
    "STABLE_ENVS_TUNING_HINT", "flag", True, "Set to 0 to hide the `stable-envs tune` hint.", _environ.parse_flag
)


def paths_of(cls) -> tuple[str, ...]:
    """Return the render and simulation paths of environment class ``cls``."""
    for c in cls.__mro__:
        if c.__name__ in RENDER_PATHS:
            return RENDER_PATHS[c.__name__]
    return RENDER_PATHS.get(getattr(cls, "family", ""), ())


def check(knobs: Mapping[str, Any]) -> dict[str, int | None]:
    """Validate knob names and values; raise ValueError for an unknown name or invalid value."""
    unknown = sorted(n for n in knobs if n not in KNOBS)
    if unknown:
        raise ValueError(f"unknown tuning knob(s) {unknown}; the knobs are {sorted(KNOBS)}")
    return {name: KNOBS[name].check(value) for name, value in knobs.items()}


# --------------------------------------------------------------------------- the measured cache


@cache
def _cpu_name() -> str:
    with contextlib.suppress(OSError), open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "cpu"


@cache
def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _dev(device) -> str | None:
    """Return ``"cpu"`` or ``"cuda:<index>"`` for ``device``, or None for CUDA when it is unavailable."""
    if not str(device).startswith("cuda"):
        return "cpu"
    if not _cuda():
        return None
    import torch

    index = torch.device(device).index
    return f"cuda:{torch.cuda.current_device() if index is None else index}"


def _where(knob: Knob, device) -> str | None:
    return _dev("cpu" if knob.device == "cpu" else device or ("cuda" if knob.device == "cuda" else "cpu"))


@cache
def stamp(device: str) -> dict[str, str]:
    """Return what the cache file of ``"cpu"`` or ``"cuda:<index>"`` must match to be used."""
    import torch

    from .__about__ import __version__

    cuda = device.startswith("cuda")
    name = torch.cuda.get_device_name(torch.device(device)) if cuda else _cpu_name()
    return {"device": name, "driver": _driver() if cuda else "", "cpu": _cpu_name(), "stable_envs": __version__}


@cache
def _driver() -> str:
    with contextlib.suppress(OSError), open("/proc/driver/nvidia/version") as fh:
        m = re.search(r"Kernel Module(?: for \S+)?\s+([\d.]+)", fh.read())
        if m:
            return m.group(1)
    import torch

    return f"cuda {torch.version.cuda}"


def cache_path(device="cpu") -> Path | None:
    """Return ``$XDG_CACHE_HOME/stable-envs/tuning/<device name>.json``, or None without CUDA for a CUDA device."""
    where = _dev(device)
    if where is None:
        return None
    slug = re.sub(r"[^a-z0-9._-]+", "-", stamp(where)["device"].lower()).strip("-") or "device"
    return _cache_dir() / "tuning" / f"{slug}.json"


def batch_range(batch: int) -> tuple[int, int]:
    """Return the power-of-two batch range ``[lo, hi]`` that shares cached values with ``batch``."""
    lo = 1 << max(0, int(batch).bit_length() - 1)
    return lo, 2 * lo - 1


def entry_key(path: str, size: int, batch: int) -> str:
    """Return the cache key of a render path at one frame size and batch range."""
    return "{}|{}|{}-{}".format(path, int(size), *batch_range(batch))


_files: dict[Path, tuple[float, dict, bool]] = {}


def read_cache(device="cpu") -> tuple[dict, bool]:
    """Return the cache file of ``device`` and whether it is current (same device, driver, CPU and version)."""
    path = cache_path(device)
    try:
        mtime = path.stat().st_mtime
    except (OSError, AttributeError):
        return {}, False
    hit = _files.get(path)
    if hit is None or hit[0] != mtime:
        try:
            data = json.loads(path.read_text())
            data["entries"].items()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            warnings.warn(f"ignoring tuning cache {path}: {e!r}", RuntimeWarning, stacklevel=2)
            data = {}
        hit = _files[path] = (mtime, data, bool(data) and data.get("stamp") == stamp(_dev(device)))
        _log("DEBUG", f"Read the tuning cache {path} (current: {hit[2]}).")
    return hit[1], hit[2]


def write_cache(entries: Mapping[str, Mapping[str, int]], device="cpu") -> Path:
    """Merge ``{key: {knob: value}}`` into the cache file of ``device`` and return its path."""
    path = cache_path(device)
    if path is None:
        raise RuntimeError("CUDA is unavailable, so there is no CUDA device to write tuning for")
    for knobs in entries.values():
        check(knobs)
    data, current = read_cache(device)
    merged = dict(data.get("entries", {}) if current else {})
    for key, knobs in entries.items():
        merged[key] = {**merged.get(key, {}), **knobs}
    where = _dev(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stamp": stamp(where), "entries": merged}, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    _files.pop(path, None)
    return path


# --------------------------------------------------------------------------- lookup

_set: dict[str, int] = {}
_scopes: contextvars.ContextVar[tuple[Mapping[str, int], ...]] = contextvars.ContextVar("tuning_scopes", default=())
_derived: dict[tuple[str, str], int | None] = {}
_MISSING = object()


def lookup(name, *, path=None, size=None, batch=None, device=None, default=_MISSING, measured=True):
    """Return ``(value, source)`` for knob ``name``; see :func:`get`."""
    if name not in KNOBS:
        check({name: None})
    knob = KNOBS[name]
    for scope in reversed(_scopes.get()):
        if name in scope:
            return scope[name], "override"
    if name in _set:
        return _set[name], "set"
    value = _environ.read(knob.env_var)
    if value is not None:
        return value, "env"
    where = _where(knob, device)
    if where is not None:
        if measured:
            data, current = read_cache(where)
            key = DEVICE_ENTRY if path is None else entry_key(path, size or 0, batch or 1)
            value = data["entries"].get(key, {}).get(name) if current else None
            if value is not None:
                return value, "cache"
        if knob.derive is not None:
            if (name, where) not in _derived:
                _derived[name, where] = knob.derive(where)
            if _derived[name, where] is not None:
                return _derived[name, where], "derived"
    return (knob.fallback, "fallback") if default is _MISSING else (default, "default")


def get(
    name: str,
    *,
    path: str | None = None,
    size: int | None = None,
    batch: int | None = None,
    device: Any = None,
    default: Any = _MISSING,
) -> int | None:
    """Return the current value of knob ``name``.

    Args:
        name: A key of :data:`KNOBS`.
        path: The render path of the call, which selects measured values together with ``size`` and
            ``batch``. None uses values measured for the whole device.
        size: Frame size in pixels; 0 for simulation paths.
        batch: Number of lanes.
        device: The device the work runs on; None uses the knob's own device.
        default: The call site's value, used instead of the fallback when nothing else sets the knob.

    Raises:
        ValueError: ``name`` is not a knob, or its environment variable is invalid.
    """
    return lookup(name, path=path, size=size, batch=batch, device=device, default=default)[0]


def set(**knobs: int | None) -> dict[str, int]:  # noqa: A001 - mirrors ``snv.set``
    """Set knobs for the process; None clears a knob. Return every knob set this way."""
    for name, value in check(knobs).items():
        if value is None:
            _set.pop(name, None)
        else:
            _set[name] = value
    return dict(_set)


def reset() -> None:
    """Clear every knob set with :func:`set`."""
    _set.clear()


@contextlib.contextmanager
def override(**knobs: int) -> Iterator[None]:
    """Set knobs inside a ``with`` block, above :func:`set`."""
    values = {k: v for k, v in check(knobs).items() if v is not None}
    token = _scopes.set((*_scopes.get(), values))
    try:
        yield
    finally:
        _scopes.reset(token)


def describe(**where) -> dict[str, tuple[int | None, str]]:
    """Return ``{name: (value, source)}`` for every knob; keywords are those of :func:`get`."""
    return {name: lookup(name, **where) for name in KNOBS}


# --------------------------------------------------------------------------- hint and autotune

_noticed: dict[tuple[str, str], None] = {}


def on_render(path: str, size: int, batch: int, env) -> None:
    """Log a one-time ``stable-envs tune`` hint, or autotune, the first time a large untuned path renders.

    Runs once per process, path and device, from 256 lanes. Never raises.
    """
    try:
        key = (path, str(env.device))
        if batch < HINT_LANES or key in _noticed:
            return
        _noticed[key] = None
        knobs = [k for k in KNOBS.values() if path in k.paths and k.unit is not None]
        if not any(k.device in ("any", env.device.type) for k in knobs):
            return
        data, current = read_cache(str(env.device))
        entries = data.get("entries", {}) if current else {}
        if entry_key(path, size, batch) in entries or any(k.name in entries.get(DEVICE_ENTRY, {}) for k in knobs):
            return
        if _environ.read("STABLE_ENVS_AUTOTUNE"):
            autotune(type(env).__name__, size, batch, str(env.device))
        elif _environ.read("STABLE_ENVS_TUNING_HINT"):
            _log(
                "INFO",
                f"`stable-envs tune {type(env).__name__} --batch {batch} --size {size} --device {env.device}` "
                f"may speed up the {path} render path (about 1 min). "
                "To hide this message, set STABLE_ENVS_TUNING_HINT=0.",
            )
    except Exception as e:  # noqa: BLE001 - a hint must never break rendering
        _log("DEBUG", f"tuning check skipped: {type(e).__name__}: {e}")


def autotune(name: str, size: int, batch: int, device: str) -> None:
    """Tune a fresh ``name`` environment once, under a lock beside the cache file; keep the defaults on contention."""
    lock = cache_path(device).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        return
    with open(lock, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _log("INFO", f"Using the derived defaults because another process is tuning {name}.")
            return
        try:
            from . import registry
            from ._tune import tune

            with registry.make(name, num_envs=batch, size=size, device=device) as env:
                tune(env, echo=None)
            _log("INFO", f"Tuned {name} and cached the values in {cache_path(device)}.")
        except Exception as e:  # noqa: BLE001 - fall back to the defaults
            _log("WARNING", f"The autotune of {name} failed ({type(e).__name__}: {e}). Using the derived defaults.")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _clear_caches() -> None:
    _files.clear()
    _derived.clear()
    _noticed.clear()
    _cuda.cache_clear()
    stamp.cache_clear()
    _environ._parsed.cache_clear()
