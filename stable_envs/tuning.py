"""Speed and memory knobs: one registry, read at call time.

A knob changes how work is split (chunk sizes, block sizes, worker counts, memory budgets); it never
changes frames or State. Knobs belong to render paths, such as ``dmc-suite`` for every fixed-camera
DMC cell. Each lookup takes the first value found in this order:

1. the per-instance ``tuning={...}`` an environment was built with, while that environment runs;
2. :func:`override` scopes, innermost first, then :func:`set`;
3. the environment variable ``STABLE_ENVS_<NAME>``;
4. a value ``stable-envs tune`` measured for the running environment's render path, resolution and
   batch range on this device, driver and stable-envs version;
5. a value derived from the hardware, for knobs that have a rule;
6. the call site's measured default, or the knob's fallback.

Nothing is measured unless ``stable-envs tune`` or :func:`stable_envs.tune` runs, or
``STABLE_ENVS_AUTOTUNE=1`` is set.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import logging
import math
import os
import platform
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._core import environ as _environ

__all__ = [
    "Knob",
    "KNOBS",
    "RENDER_PATHS",
    "get",
    "set",
    "reset",
    "override",
    "describe",
    "paths_of",
    "cache_path",
    "device_name",
]


def l3_bytes() -> int | None:
    """Return the size of the L3 cache the first CPU shares, or None where it is not readable."""
    try:
        size = os.sysconf("SC_LEVEL3_CACHE_SIZE")
        if size > 0:
            return int(size)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        for index in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
            if (index / "level").read_text().strip() == "3":
                text = (index / "size").read_text().strip().upper()
                scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(text[-1:], 1)
                return int(text.rstrip("KMG")) * scale
    except (OSError, ValueError):
        pass
    return None


def free_cuda_bytes(device) -> int | None:
    """Return the free memory of CUDA ``device``, or None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.mem_get_info(device)[0])


def _pow2_floor(x: float) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(1.0, x)))))


# The CPU ray budget was measured with a 35.75 MiB L3 (Xeon Platinum 8268).
_MEASURED_L3 = 36608 << 10
#: Estimated intermediate bytes per ray of the fixed-camera DMC renderer.
DMC_BYTES_PER_RAY = 256


def _derive_cpu_ray_budget(device) -> int | None:
    l3 = l3_bytes()
    if l3 is None:
        return None
    return 1 << min(21, max(18, round(math.log2((1 << 21) * l3 / _MEASURED_L3))))


def _derive_cuda_ray_budget(device) -> int | None:
    free = free_cuda_bytes(device)
    return None if free is None else min(1 << 24, _pow2_floor(free / (2 * DMC_BYTES_PER_RAY)))


def _derive_nvd_budget(device) -> int | None:
    free = free_cuda_bytes(device)
    return None if free is None else min(6 << 30, free // 2)


@dataclass(frozen=True)
class Knob:
    """One tuning knob.

    Attributes:
        name: Knob name; the environment variable is ``STABLE_ENVS_`` plus the name in upper case.
        fallback: Value used when no layer sets one. None means the call site decides
            (a measured per-case table, or a rule that needs the device).
        doc: What the knob controls.
        paths: Render paths (keys of :data:`RENDER_PATHS` values) that read the knob.
        device: ``"cpu"``, ``"cuda"`` or ``"any"``: the hardware the knob's work runs on.
        minimum: Smallest accepted value.
        power_of_two: Whether a value must be a power of two (Triton block sizes).
        choices: Accepted values, when the set is small (Triton warp counts).
        derive: Rule computing a value from the hardware, or None.
        derived: How ``derive`` computes the value, for the documentation.
        measured_on: The machine the fallback was measured on, where recorded.
        unit: What one unit of the value covers, for pruning the sweep: ``"rows"`` (batch rows),
            ``"rays"`` (pixels times supersamples), ``"cpus"``, ``"bytes"`` or ``""``.
        tunable: Whether ``tune`` sweeps the knob.
        rebuild: Whether the knob is read when an environment is built, so ``tune`` rebuilds it.
        workload: What ``tune`` times for this knob: ``"render"`` or ``"step"``.
    """

    name: str
    fallback: int | None
    doc: str
    paths: tuple[str, ...]
    device: str = "any"
    minimum: int = 1
    power_of_two: bool = False
    choices: tuple[int, ...] = ()
    derive: Callable[[Any], int | None] | None = None
    derived: str = ""
    measured_on: str = ""
    unit: str = ""
    tunable: bool = True
    rebuild: bool = False
    workload: str = "render"

    @property
    def env_var(self) -> str:
        """The environment variable that sets this knob."""
        return f"STABLE_ENVS_{self.name.upper()}"

    def check(self, value: Any) -> int | None:
        """Return ``value`` as a validated int, or None to leave the knob unset.

        Raises:
            ValueError: The value is not an int, is below ``minimum``, is not a power of two where
                one is required, or is not one of ``choices``.
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"tuning knob {self.name!r} takes an int, got {value!r}")
        if value < self.minimum:
            raise ValueError(f"tuning knob {self.name!r} must be at least {self.minimum}, got {value}")
        if self.power_of_two and value & (value - 1):
            raise ValueError(f"tuning knob {self.name!r} must be a power of two, got {value}")
        if self.choices and value not in self.choices:
            raise ValueError(f"tuning knob {self.name!r} must be one of {list(self.choices)}, got {value}")
        return int(value)


#: Render and simulation paths per environment class or family name; the most derived class wins.
RENDER_PATHS: dict[str, tuple[str, ...]] = {
    "DMControlSuite": ("dmc-suite", "dmc-sim"),
    "DMControlTrackCom": ("dmc-trackcom", "dmc-sim"),
    "dmcontrol": ("dmc-sim",),
    "OGBManipBase": ("ogb-manipulation", "ogb-sim"),
    "ogbench": ("ogb-maze", "ogb-sim"),
    "fetch": ("fetch-mjwarp", "fetch-sim"),
    "PyFlytRocketLanding": ("rocket-nvdiff",),
    "navigation": ("navigation",),
    "pusht": ("raster2d",),
    "classic_control": ("gfxdraw", "classic-sim"),
}

_WARPS = (1, 2, 4, 8, 16, 32)
_SIM = ("classic-sim", "dmc-sim", "fetch-sim", "ogb-sim")

KNOBS: dict[str, Knob] = {
    k.name: k
    for k in (
        Knob(
            "workers",
            8,
            "Most worker processes a CPU simulator batch uses on Linux; 0 steps every lane in this process.",
            _SIM,
            device="cpu",
            minimum=0,
            derived="min(value, CPUs in this process's affinity, lanes)",
            unit="cpus",
            rebuild=True,
            workload="step",
        ),
        Knob(
            "worker_min_lanes",
            64,
            "Smallest batch that uses worker processes, or Fetch simulator threads.",
            _SIM,
            device="cpu",
            tunable=False,
            rebuild=True,
            workload="step",
        ),
        Knob(
            "fetch_threads",
            4,
            "Most simulator threads of an in-process Fetch batch of at least worker_min_lanes lanes.",
            ("fetch-sim",),
            device="cpu",
            derived="min(value, CPUs in this process's affinity, lanes)",
            unit="cpus",
            rebuild=True,
            workload="step",
        ),
        Knob(
            "cpu3d_chunk",
            16,
            "Frames per cpu3d ray-casting chunk. TrackCom cells use their measured per-size table when unset.",
            ("dmc-trackcom",),
            device="cpu",
            measured_on="Xeon Platinum 8268 (TrackCom per-size table)",
            unit="rows",
        ),
        Knob(
            "dmc_cpu_ray_budget",
            1 << 21,
            "Rays per CPU render chunk of the fixed-camera DMC cells, counting supersamples.",
            ("dmc-suite",),
            device="cpu",
            power_of_two=True,
            derive=_derive_cpu_ray_budget,
            derived="2^21 scaled by L3 size / 35.75 MiB, rounded to a power of two, in [2^18, 2^21]",
            unit="rays",
        ),
        Knob(
            "dmc_ray_budget",
            1 << 24,
            "Rays per CUDA render chunk of the fixed-camera DMC cells, counting supersamples.",
            ("dmc-suite",),
            device="cuda",
            power_of_two=True,
            derive=_derive_cuda_ray_budget,
            derived="min(2^24, free GPU memory / 512, rounded down to a power of two)",
            unit="rays",
        ),
        Knob(
            "dmc_launch_rays",
            256 * 224 * 224,
            "Rays per MJWarp launch of the DMC TrackCom cells; sets worlds per launch (at most 256).",
            ("dmc-trackcom",),
            device="cuda",
            unit="rays",
            rebuild=True,
        ),
        Knob(
            "dmc_min_worlds",
            64,
            "Fewest worlds per MJWarp launch of the DMC TrackCom cells.",
            ("dmc-trackcom",),
            device="cuda",
            power_of_two=True,
            unit="rows",
            rebuild=True,
        ),
        Knob(
            "fetch_mjw_worlds",
            256,
            "Most worlds per MJWarp launch of the Fetch cells.",
            ("fetch-mjwarp",),
            device="cuda",
            power_of_two=True,
            unit="rows",
        ),
        Knob(
            "ogb_mjw_worlds",
            256,
            "Most worlds per MJWarp launch of the OGBench Cube and Scene cells.",
            ("ogb-manipulation",),
            device="cuda",
            power_of_two=True,
            unit="rows",
        ),
        Knob(
            "ogb_context_cache_bytes",
            None,
            "GPU bytes the OGBench MJWarp context cache keeps resident per device.",
            ("ogb-manipulation",),
            device="cuda",
            derived="min(total GPU memory / 4, free GPU memory / 2), per device",
            tunable=False,
        ),
        Knob(
            "nvd_budget_bytes",
            6 << 30,
            "GPU bytes one nvdiffrast chunk of the Rocket cells may use; sets worlds per chunk.",
            ("rocket-nvdiff",),
            device="cuda",
            derive=_derive_nvd_budget,
            derived="min(6 GiB, free GPU memory / 2)",
            unit="bytes",
        ),
        Knob(
            "nav_render_chunk",
            256,
            "Frames per render chunk of the navigation cells.",
            ("navigation",),
            unit="rows",
        ),
        Knob(
            "navdot_block",
            1024,
            "Pixels per Triton tile of the navigation dot compositor.",
            ("navigation",),
            device="cuda",
            power_of_two=True,
            minimum=16,
        ),
        Knob(
            "navdot_warps",
            4,
            "Triton warps per tile of the navigation dot compositor.",
            ("navigation",),
            device="cuda",
            choices=_WARPS,
        ),
        Knob(
            "raster2d_block",
            128,
            "Pixels per Triton program of the 2-D rasterizer when it does not cull.",
            ("raster2d", "navigation"),
            device="cuda",
            power_of_two=True,
            minimum=16,
        ),
        Knob(
            "raster2d_cull_warps",
            4,
            "Triton warps per tile of the 2-D rasterizer when it culls.",
            ("raster2d", "navigation"),
            device="cuda",
            choices=_WARPS,
        ),
        Knob(
            "gfxdraw_warps",
            1,
            "Triton warps per tile of the pygame-compatible gfxdraw rasterizer.",
            ("gfxdraw",),
            device="cuda",
            choices=_WARPS,
        ),
        Knob(
            "raycast_block",
            256,
            "Rays per Triton program of the analytic ray caster.",
            ("dmc-suite", "ogb-maze"),
            device="cuda",
            power_of_two=True,
            minimum=16,
        ),
    )
}


def paths_of(cls) -> tuple[str, ...]:
    """Return the render and simulation paths of environment class ``cls``, most specific first."""
    for c in cls.__mro__:
        if c.__name__ in RENDER_PATHS:
            return RENDER_PATHS[c.__name__]
    return RENDER_PATHS.get(getattr(cls, "family", ""), ())


_set: dict[str, int] = {}
_scopes: contextvars.ContextVar[tuple[Mapping[str, int], ...]] = contextvars.ContextVar(
    "stable_envs_tuning_scopes", default=()
)
# The running environment: (its instance knobs, the env, a render size override).
_running: contextvars.ContextVar[tuple[Mapping[str, int] | None, Any, int | None] | None] = contextvars.ContextVar(
    "stable_envs_tuning_running", default=None
)
_MISSING = object()


def _knob(name: str) -> Knob:
    try:
        return KNOBS[name]
    except KeyError:
        raise ValueError(f"unknown tuning knob {name!r}; the knobs are {sorted(KNOBS)}") from None


def check(knobs: Mapping[str, Any]) -> dict[str, int | None]:
    """Validate a mapping of knob names to values and return it as a dict.

    Raises:
        ValueError: A name is not a knob or a value is invalid.
    """
    return {name: _knob(name).check(value) for name, value in knobs.items()}


for _knob_entry in KNOBS.values():
    _environ.register(
        _knob_entry.env_var,
        "bytes" if _knob_entry.name.endswith("_bytes") else "int",
        None,
        f"Tuning knob `{_knob_entry.name}`: {_knob_entry.doc}",
        lambda text, knob=_knob_entry: knob.check(_environ.parse_int(text)),
    )
del _knob_entry
_environ.register(
    "STABLE_ENVS_AUTOTUNE",
    "flag",
    False,
    "1 tunes an untuned render path the first time it renders a large batch, then caches the values.",
    _environ.parse_flag,
)
_environ.register(
    "STABLE_ENVS_TUNING_HINT",
    "flag",
    True,
    "0 silences the one-time log line that suggests `stable-envs tune` for a large untuned render path.",
    _environ.parse_flag,
)


# --------------------------------------------------------------------------- the measured cache


@functools.cache
def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "cpu"


@functools.cache
def _cuda_name(index: int) -> str:
    import torch

    return torch.cuda.get_device_name(index)


@functools.cache
def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _cuda_index(device) -> int | None:
    import torch

    d = torch.device(device)
    if d.type != "cuda" or not _cuda_available():
        return None
    return torch.cuda.current_device() if d.index is None else d.index


def device_name(device=None) -> str | None:
    """Return the hardware name the cache file of ``device`` is keyed by.

    Args:
        device: A torch device or its string. None or a CPU device names the CPU model; a CUDA device
            without an index names the current CUDA device.

    Returns:
        The name, or None for a CUDA device when CUDA is unavailable.
    """
    if device is None or str(device).split(":")[0] != "cuda":
        return _cpu_name()
    index = _cuda_index(device)
    return None if index is None else _cuda_name(index)


@functools.cache
def _driver() -> str:
    try:
        with open("/proc/driver/nvidia/version") as fh:
            m = re.search(r"Kernel Module(?: for \S+)?\s+([\d.]+)", fh.read())
            if m:
                return m.group(1)
    except OSError:
        pass
    import torch

    return f"cuda {torch.version.cuda}"


def stamp(device=None) -> dict[str, str]:
    """Return what a cache file must match to be used: device, driver, host CPU and stable-envs version."""
    from .__about__ import __version__

    cuda = str(device).split(":")[0] == "cuda"
    return {
        "device": device_name(device) or "",
        "driver": _driver() if cuda else "",
        "cpu": _cpu_name(),
        "stable_envs": __version__,
    }


@functools.lru_cache(maxsize=64)
def _path(name: str, base: str) -> Path:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "device"
    return Path(base) / "stable-envs" / "tuning" / f"{slug}.json"


def cache_path(device=None) -> Path | None:
    """Return the tuning cache file for ``device``, or None for CUDA when CUDA is unavailable.

    The file is ``$XDG_CACHE_HOME/stable-envs/tuning/<device name>.json`` (``~/.cache`` when
    ``XDG_CACHE_HOME`` is unset), with the name lower-cased and reduced to letters, digits, ``.``,
    ``_`` and ``-``.
    """
    name = device_name(device)
    if name is None:
        return None
    return _path(name, os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache"))


def batch_range(batch: int) -> tuple[int, int]:
    """Return the power-of-two range ``[lo, hi]`` of batch sizes that share cached values with ``batch``."""
    lo = 1 << max(0, int(batch).bit_length() - 1)
    return lo, 2 * lo - 1


def entry_key(path: str, size: int, batch: int) -> str:
    """Return the cache key of a render path at one resolution and batch range."""
    lo, hi = batch_range(batch)
    return f"{path}|{int(size)}|{lo}-{hi}"


_files: dict[Path, tuple[float, dict]] = {}


def read_cache(device=None) -> tuple[dict, bool]:
    """Return the cache file of ``device`` as ``(data, current)``; ``current`` is False when it is stale."""
    path = cache_path(device)
    if path is None:
        return {}, False
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}, False
    hit = _files.get(path)
    if hit is None or hit[0] != mtime:
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("entries", {}), dict):
                raise ValueError("not a tuning cache")
        except (OSError, ValueError) as e:
            warnings.warn(f"ignoring tuning cache {path}: {e}", RuntimeWarning, stacklevel=2)
            data = {}
        _files[path] = hit = (mtime, data)
    data = hit[1]
    return data, bool(data) and data.get("stamp") == stamp(device)


def _measured(name: str, device, env, size: int | None) -> int | None:
    batch = getattr(env, "B", None)
    size = size or getattr(env, "size", None)
    if batch is None or size is None:
        return None
    data, current = read_cache(device)
    if not current:
        return None
    entries = data.get("entries", {})
    for path in paths_of(type(env)):
        value = entries.get(entry_key(path, size, batch), {}).get("knobs", {}).get(name)
        if value is not None:
            try:
                return KNOBS[name].check(value)
            except ValueError:
                return None
    return None


def write_cache(entries: Mapping[str, Mapping[str, Any]], device=None) -> Path:
    """Merge ``entries`` (cache key to ``{"knobs": ..., ...}``) into the cache file of ``device``."""
    path = cache_path(device)
    if path is None:
        raise RuntimeError("CUDA is unavailable, so there is no CUDA device to write tuning for")
    for entry in entries.values():
        check(entry["knobs"])
    data, current = read_cache(device)
    data = {"stamp": stamp(device), "entries": {**(data.get("entries", {}) if current else {}), **entries}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    _files.pop(path, None)
    return path


# --------------------------------------------------------------------------- lookup


def _device(knob: Knob, device, env) -> str:
    """Return the device string whose cache file and hardware ``knob`` reads."""
    if device is None and env is not None and knob.device != "cpu":
        device = getattr(env, "device", None)
    if knob.device == "cpu":
        return "cpu"
    if device is None:
        return "cuda" if knob.device == "cuda" else "cpu"
    d = str(device)
    return d if d.startswith("cuda") else ("cuda" if knob.device == "cuda" else "cpu")


_derived: dict[tuple[str, str], int | None] = {}


def _resolve(name: str, device=None, default: Any = _MISSING, *, measured: bool = True) -> tuple[int | None, str]:
    knob = _knob(name)
    current = _running.get()
    inst, env, size = current if current is not None else (None, None, None)
    if inst and name in inst:
        return inst[name], "instance"
    for scope in reversed(_scopes.get()):
        if name in scope:
            return scope[name], "override"
    if name in _set:
        return _set[name], "set"
    value = _environ.read(knob.env_var)
    if value is not None:
        return value, "env"
    where = _device(knob, device, env)
    cuda = where.startswith("cuda")
    if measured and env is not None and (not cuda or _cuda_available()):
        value = _measured(name, where, env, size)
        if value is not None:
            return value, "cache"
    if knob.derive is not None and (not cuda or _cuda_available()):
        key = (name, where if not cuda else f"cuda:{_cuda_index(where)}")
        if key not in _derived:
            _derived[key] = knob.derive(key[1])
        if _derived[key] is not None:
            return _derived[key], "derived"
    if default is not _MISSING:
        return default, "default"
    return knob.fallback, "fallback"


def get(name: str, *, device=None, default: Any = _MISSING) -> int | None:
    """Return the current value of knob ``name``.

    Args:
        name: A key of :data:`KNOBS`.
        device: The device the work runs on; it selects the cache file and the hardware a derived
            value reads. None uses the running environment's device, or the knob's own device.
        default: Value to use instead of the knob's fallback when no layer sets one. Call sites
            with a measured per-case value pass it here.

    Raises:
        ValueError: ``name`` is not a knob, or its environment variable is invalid.
    """
    return _resolve(name, device, default)[0]


def set(**knobs: int | None) -> dict[str, int]:  # noqa: A001 - mirrors ``snv.set``
    """Set knobs for the whole process; a value of None clears that knob.

    Returns:
        A copy of every knob set this way.

    Raises:
        ValueError: A name is not a knob or a value is invalid; nothing is changed then.
    """
    values = check(knobs)
    for name, value in values.items():
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
    """Set knobs inside a ``with`` block, above :func:`set` and below an environment's own ``tuning``.

    Raises:
        ValueError: A name is not a knob or a value is invalid.
    """
    values = {k: v for k, v in check(knobs).items() if v is not None}
    token = _scopes.set((*_scopes.get(), values))
    try:
        yield
    finally:
        _scopes.reset(token)


@contextlib.contextmanager
def running(env, size: int | None = None) -> Iterator[None]:
    """Mark ``env`` as the running environment: its own knobs apply and its render path selects cached values."""
    current = _running.get()
    if current is not None and current[1] is env and (size is None or size == current[2]):
        yield
        return
    token = _running.set((env.__dict__.get("_tuning"), env, size))
    try:
        yield
    finally:
        _running.reset(token)


def is_running(env) -> bool:
    """Whether ``env`` is the running environment of this context."""
    current = _running.get()
    return current is not None and current[1] is env


#: Batch size from which an untuned render path logs the ``stable-envs tune`` hint or autotunes.
HINT_LANES = 256
logger = logging.getLogger("stable_envs.tuning")
_noticed: set[tuple[str, str]] = set()


@functools.lru_cache(maxsize=256)
def _tunable_path(cls, kind: str) -> str | None:
    """The first path of ``cls`` with a knob ``tune`` sweeps on ``kind`` devices."""
    for p in paths_of(cls):
        if any(k.tunable and p in k.paths and k.device in ("any", kind) for k in KNOBS.values()):
            return p
    return None


def on_render(env, size: int | None = None, *, may_tune: bool = True) -> None:
    """Log a one-time ``stable-envs tune`` hint, or autotune, the first time a large untuned path renders.

    ``STABLE_ENVS_AUTOTUNE=1`` tunes instead of logging when the render is not inside another call of
    the environment; ``STABLE_ENVS_TUNING_HINT=0`` silences the hint. Nothing here raises into the render.
    """
    try:
        if getattr(env, "B", 0) < HINT_LANES:
            return
        path = _tunable_path(type(env), env.device.type)
        if path is None or (path, str(env.device)) in _noticed:
            return
        data, current = read_cache(str(env.device))
        if current and entry_key(path, size or env.size, env.B) in data.get("entries", {}):
            return
        autotuning = _environ.read("STABLE_ENVS_AUTOTUNE") and not env.__dict__.get("_tuning")
        if autotuning and not may_tune:
            return
        _noticed.add((path, str(env.device)))
        if autotuning:
            autotune(env, path)
        elif _environ.read("STABLE_ENVS_TUNING_HINT"):
            logger.info(
                "`stable-envs tune %s --batch %d --device %s` may speed up the %s render path (about 1 min)",
                env.class_name,
                env.B,
                env.device,
                path,
            )
    except Exception as e:  # noqa: BLE001 - a hint must never break rendering
        logger.info("tuning check skipped: %s: %s", type(e).__name__, e)


@contextlib.contextmanager
def _try_lock(path: Path) -> Iterator[bool]:
    """Hold an exclusive lock on ``path`` without waiting; yield False when another process holds it."""
    try:
        import fcntl
    except ImportError:
        yield False
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def autotune(env, path: str) -> None:
    """Tune ``env`` once under a lock beside its cache file; on contention or failure keep the defaults."""
    target = cache_path(env.device)
    if target is None:
        return
    with _try_lock(target.with_suffix(".lock")) as held:
        if not held:
            logger.info("another process is tuning %s; using derived defaults", path)
            return
        try:
            from ._tune import tune

            tune(env, echo=None)
            logger.info("tuned the %s render path on %s and cached it in %s", path, device_name(env.device), target)
        except Exception as e:  # noqa: BLE001 - fall back to the defaults
            logger.info("autotune of %s failed (%s: %s); using derived defaults", path, type(e).__name__, e)


def describe(device=None) -> dict[str, dict[str, Any]]:
    """Return every knob's current value and the layer it comes from.

    Args:
        device: As in :func:`get`.

    Returns:
        ``{name: {"value", "source", "fallback", "env_var", "paths", "doc"}}``; ``source`` is one of
        ``instance``, ``override``, ``set``, ``env``, ``cache``, ``derived`` or ``fallback``.
    """
    out = {}
    for name, knob in KNOBS.items():
        value, source = _resolve(name, device)
        out[name] = dict(
            value=value, source=source, fallback=knob.fallback, env_var=knob.env_var, paths=knob.paths, doc=knob.doc
        )
    return out


def _clear_caches() -> None:
    """Forget memoized cache files and derived values, as after a hardware or file change."""
    _files.clear()
    _derived.clear()
    _environ._parsed.cache_clear()
    _cuda_available.cache_clear()
    _noticed.clear()
