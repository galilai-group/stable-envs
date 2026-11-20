"""Tuning settings: chunk sizes, block sizes, worker counts and memory budgets.

A setting changes speed and memory use, but frames and state are the same for every valid value.
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

__all__ = [
    "Setting",
    "SETTINGS",
    "RENDER_PATHS",
    "get",
    "set",
    "reset",
    "override",
    "describe",
    "cache_path",
    "render_paths",
    "time_backend",
]

#: Batch size from which an untuned render path logs the ``stable-envs tune`` hint or autotunes.
HINT_MIN_ENVS = 256
#: Cache entry of the settings read without a render path (Triton kernel shapes).
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
class Setting:
    """One setting: its fallback, the render paths that read it, and how ``tune`` sweeps it.

    Attributes:
        name: Setting name, a key of ``SETTINGS``.
        fallback: Value used when nothing else sets the setting (the call site decides if None).
        paths: Render or simulation paths that read the setting.
        device: Device whose work the setting controls: ``"cpu"``, ``"cuda"`` or ``"any"``.
        kind: Accepted values: ``"n"`` (at least 1), ``"n0"`` (at least 0), ``"pow2"`` or ``"warps"``.
        unit: Work-split unit ``tune`` uses to skip equivalent values (not swept if None).
        doc: One-line description.
        derive: ``device -> value`` hardware rule.
        step: Read at construction and timed by stepping.
        per_path: Store measured values per render path instead of per device.
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
        """The environment variable that sets this setting."""
        return f"STABLE_ENVS_{self.name.upper()}"

    def check(self, value: Any) -> int | None:
        """Return ``value`` unchanged if it is None or valid for this setting's ``kind``.

        Args:
            value: The value to check.

        Raises:
            ValueError: If ``value`` is a bool, not an int, or outside the values ``kind`` allows.
        """
        if value is None:
            return None
        ok = isinstance(value, int) and not isinstance(value, bool) and value >= (self.kind != "n0")
        if ok and self.kind == "pow2":
            ok = value & (value - 1) == 0
        if ok and self.kind == "warps":
            ok = value in _WARPS
        if not ok:
            raise ValueError(f"tuning setting {self.name!r} takes {_KINDS[self.kind]}, got {value!r}")
        return value


_KINDS = {"n": "an int >= 1", "n0": "an int >= 0", "pow2": "a power of two", "warps": f"one of {list(_WARPS)}"}
_SIM = (
    "classic-sim",
    "fetch-sim",
    "ogb-sim",
    "pusht-sim",
    "pusht-render",
    "rocket-sim",
)
_RENDER = ("classic-render", "fetch-opengl", "miniworld-render", "roadster-render", "rocket-render")

SETTINGS: dict[str, Setting] = {
    k.name: k
    for k in (
        Setting("workers", 8, _SIM, "cpu", "n0", "cpus", "Most worker processes of a CPU simulator batch or PushT render on Linux; 0 runs in-process.", step=True),
        Setting("large_renderer_min_rows", None, (), "any", "n", None, "Smallest render batch that draws with the GPU renderer of a class that has a CPU and a GPU renderer; smaller batches draw on the CPU and move the frames to the device in one copy; stable-envs tune measures it, else each class ships a value."),
        Setting("worker_min_envs", 64, _SIM, "cpu", "n", None, "Smallest batch that uses worker processes or Fetch threads."),
        Setting("worker_timeout", 120, _SIM, "cpu", "n", None, "Seconds a CPU simulator worker may take to answer one request before the batch fails; starting takes up to the larger of this and 300.", per_path=False),
        Setting("render_workers", 8, _RENDER, "cpu", "n0", "cpus", "Most worker processes of a CPU render batch on Linux; 0 or 1 draws in-process."),
        Setting("render_worker_min_rows", 16, _RENDER, "cpu", "n", None, "Smallest CPU render batch that uses render worker processes."),
        Setting("fetch_threads", 4, ("fetch-sim",), "cpu", "n", "cpus", "Most simulator threads of an in-process Fetch batch.", step=True),
        Setting("dmc_threads", 8, ("dmc-sim",), "cpu", "n", "cpus", "Most mujoco.batch threads of a DM Control batch.", step=True),
        Setting("cpu3d_chunk", 16, ("dmc-trackcom",), "cpu", "n", "rows", "Frames per cpu3d ray-casting chunk."),
        Setting("dmc_cpu_ray_budget", 1 << 21, ("dmc-suite",), "cpu", "pow2", "rays", "Rays per CPU render chunk of fixed-camera DMC cells.", _cpu_rays),
        Setting("dmc_ray_budget", 1 << 24, ("dmc-suite",), "cuda", "pow2", "rays", "Rays per CUDA render chunk of fixed-camera DMC cells.", _cuda_rays),
        Setting("dmc_graph_rows", 48, ("dmc-suite",), "cuda", "n", "rows", "Most rows of a fixed-camera DMC CUDA render chunk that replay one captured graph; larger chunks launch kernels and draw only the rays movers reach."),
        Setting("dmc_launch_rays", 256 * 224 * 224, ("dmc-trackcom",), "cuda", "n", "rays", "Rays per MJWarp launch of DMC TrackCom cells."),
        Setting("dmc_min_worlds", 64, ("dmc-trackcom",), "cuda", "pow2", "rows", "Fewest worlds per MJWarp launch of DMC TrackCom cells."),
        Setting("fetch_mjw_worlds", 256, ("fetch-mjwarp",), "cuda", "pow2", "rows", "Most worlds per MJWarp launch of Fetch cells."),
        Setting("ogb_mjw_worlds", 256, ("ogb-manipulation",), "cuda", "pow2", "rows", "Most worlds per MJWarp launch of OGBench Cube and Scene.", per_path=False),
        Setting("ogb_gl_workers", 8, ("ogb-manipulation",), "any", "n0", None, "Most OpenGL worker processes of an OGBench Cube and Scene batch on Linux; 0 draws in-process.", per_path=False),
        Setting("ogb_context_cache_bytes", None, ("ogb-manipulation",), "cuda", "n", None, "GPU bytes the OGBench context cache keeps per device."),
        Setting("nvd_budget_bytes", 6 << 30, ("rocket-nvdiff",), "cuda", "n", "bytes", "GPU bytes per nvdiffrast chunk of Rocket cells.", _nvd_bytes),
        Setting("nav_render_chunk", 256, ("navigation",), "any", "n", "rows", "Frames per render chunk of navigation cells."),
        Setting("navdot_block", 1024, ("navigation",), "cuda", "pow2", "", "Pixels per Triton tile of the navigation dot compositor.", per_path=False),
        Setting("navdot_warps", 4, ("navigation",), "cuda", "warps", "", "Triton warps per tile of the navigation dot compositor.", per_path=False),
        Setting("raster2d_block", 128, ("raster2d",), "cuda", "pow2", "", "Pixels per Triton program of the unculled 2-D rasterizer.", per_path=False),
        Setting("raster2d_cull_warps", 4, ("raster2d",), "cuda", "warps", "", "Triton warps per tile of the culling 2-D rasterizer.", per_path=False),
        Setting("gfxdraw_warps", 1, ("gfxdraw",), "cuda", "warps", "", "Triton warps per tile of the gfxdraw rasterizer.", per_path=False),
        Setting("raycast_block", 256, ("dmc-suite", "ogb-maze"), "cuda", "pow2", "", "Rays per Triton program of the analytic ray caster.", per_path=False),
    )
}  # fmt: skip

#: Render and simulation paths per environment class or family; the most derived class wins.
RENDER_PATHS: dict[str, tuple[str, ...]] = {
    "DMControlSuite": ("dmc-suite", "dmc-render", "dmc-sim"),
    "DMControlTrackCom": ("dmc-trackcom", "dmc-render", "dmc-sim"),
    "OGBManipBase": ("ogb-manipulation", "ogb-sim"),
    "ogbench": ("ogb-maze", "ogb-sim"),
    "fetch": ("fetch-mjwarp", "fetch-opengl", "fetch-sim"),
    "PyFlytRocketLanding": ("rocket-nvdiff", "rocket-render", "rocket-sim"),
    "navigation": ("navigation", "raster2d"),
    "pusht": ("raster2d", "pusht-sim", "pusht-render"),
    "classic_control": ("gfxdraw", "classic-render", "classic-sim"),
    "miniworld": ("miniworld-render",),
    "roadster": ("roadster-render",),
}

for _k in SETTINGS.values():
    _environ.register(
        _k.env_var, _k.kind, None, f"Tuning setting `{_k.name}`.", lambda t, k=_k: k.check(_environ.parse_int(t))
    )
_environ.register(
    "STABLE_ENVS_AUTOTUNE", "flag", False, "1 tunes a large untuned render path on first use.", _environ.parse_flag
)
_environ.register(
    "STABLE_ENVS_TUNING_HINT", "flag", True, "Set to 0 to hide the `stable-envs tune` hint.", _environ.parse_flag
)


def paths_of(cls) -> tuple[str, ...]:
    """Return the render and simulation paths of environment class ``cls`` (empty if unknown)."""
    for c in cls.__mro__:
        if c.__name__ in RENDER_PATHS:
            return RENDER_PATHS[c.__name__]
    return RENDER_PATHS.get(getattr(cls, "family", ""), ())


def check(settings: Mapping[str, Any]) -> dict[str, int | None]:
    """Validate ``{setting name: value or None}`` and return the checked values.

    Args:
        settings: The values to check.

    Raises:
        ValueError: If a name is not in ``SETTINGS`` or a value is invalid.
    """
    unknown = sorted(n for n in settings if n not in SETTINGS)
    if unknown:
        raise ValueError(f"unknown tuning setting(s) {unknown}; the settings are {sorted(SETTINGS)}")
    return {name: SETTINGS[name].check(value) for name, value in settings.items()}


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
    """Return ``"cpu"`` or ``"cuda:<index>"`` for ``device``; None for a CUDA device this host doesn't have."""
    if not str(device).startswith("cuda"):
        return "cpu"
    if not _cuda():
        return None
    import torch

    index = torch.device(device).index
    if index is not None and index >= torch.cuda.device_count():
        return None
    return f"cuda:{torch.cuda.current_device() if index is None else index}"


def _where(setting: Setting, device) -> str | None:
    return _dev("cpu" if setting.device == "cpu" else device or ("cuda" if setting.device == "cuda" else "cpu"))


@cache
def stamp(device: str) -> dict[str, str]:
    """Return what the cache file of ``"cpu"`` or ``"cuda:<index>"`` must match to be used."""
    import torch

    from .__about__ import __version__

    cuda = device.startswith("cuda")
    name = torch.cuda.get_device_name(torch.device(device)) if cuda else _cpu_name()
    from ._core.cpus import cpu_limit

    return {
        "device": name,
        "driver": _driver() if cuda else "",
        "cpu": _cpu_name(),
        "cpus": str(cpu_limit()),
        "stable_envs": __version__,
    }


@cache
def _driver() -> str:
    with contextlib.suppress(OSError), open("/proc/driver/nvidia/version") as fh:
        m = re.search(r"Kernel Module(?: for \S+)?\s+([\d.]+)", fh.read())
        if m:
            return m.group(1)
    import torch

    return f"cuda {torch.version.cuda}"


def cache_path(device="cpu") -> Path | None:
    """Return the tuning cache file of ``device``, or None for a CUDA device without CUDA.

    The file is ``$XDG_CACHE_HOME/stable-envs/tuning/<device name>.json`` (``~/.cache`` if unset).

    Args:
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``.
    """
    where = _dev(device)
    if where is None:
        return None
    return _cache_file(where, os.environ.get("XDG_CACHE_HOME", ""))


@cache
def _cache_file(where: str, base: str) -> Path:
    """The cache file of ``where`` while ``$XDG_CACHE_HOME`` is ``base``.

    One Path per pair, so that each ``read_cache`` call reuses its string form.
    """
    slug = re.sub(r"[^a-z0-9._-]+", "-", stamp(where)["device"].lower()).strip("-") or "device"
    return _cache_dir() / "tuning" / f"{slug}.json"


def batch_range(batch: int) -> tuple[int, int]:
    """Return the power-of-two batch range ``[lo, hi]`` that shares cached values with ``batch``."""
    lo = 1 << max(0, int(batch).bit_length() - 1)
    return lo, 2 * lo - 1


def entry_key(path: str, size: int, batch: int) -> str:
    """Return the cache key of a render path at one frame size and batch range.

    Args:
        path: Render or simulation path.
        size: Frame size in pixels (0 for simulation paths).
        batch: Number of environments; it's rounded to its power-of-two range.
    """
    return "{}|{}|{}-{}".format(path, int(size), *batch_range(batch))


_files: dict[Path, tuple[int, float | None, dict, bool]] = {}
#: Bumped by ``write_cache``, ``set``, ``reset``, ``override`` and ``refresh``. ``read_cache`` reads a file again only
#: after a bump, so a lookup on the step or render path costs no file system call.
_generation = 0


def _bump() -> None:
    global _generation
    _generation += 1


def refresh() -> None:
    """Read the cache files again at their next use. Building an environment calls it."""
    _bump()


def read_cache(device="cpu") -> tuple[dict, bool]:
    """Return the cache file of ``device`` and whether it is current (same device, driver, CPU and version).

    The file is read again only after ``refresh`` or a change of settings in this process. A file that another process
    writes is read when this process builds its next environment.

    Args:
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``.
    """
    path = cache_path(device)
    if path is None:
        return {}, False
    generation = _generation
    hit = _files.get(path)
    if hit is not None and hit[0] == generation:
        return hit[2], hit[3]
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if mtime is None:
        data, current = {}, False
    elif hit is not None and hit[1] == mtime:
        data, current = hit[2], hit[3]
    else:
        try:
            data = json.loads(path.read_text())
            data["entries"].items()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            warnings.warn(f"ignoring tuning cache {path}: {e!r}", RuntimeWarning, stacklevel=2)
            data = {}
        current = bool(data) and data.get("stamp") == stamp(_dev(device))
        _log("DEBUG", f"Read the tuning cache {path} (current: {current}).")
    _files[path] = (generation, mtime, data, current)
    return data, current


def write_cache(entries: Mapping[str, Mapping[str, int]], device="cpu") -> Path:
    """Merge ``{entry key: {setting: value}}`` into the cache file of ``device`` and return its path.

    Entries of a file that isn't current for this machine are dropped first.

    Args:
        entries: ``{entry key: {setting: value}}``, with keys from ``entry_key`` or ``"*"`` for the whole device.
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``.

    Raises:
        RuntimeError: If ``device`` is a CUDA device and CUDA is unavailable.
    """
    path = cache_path(device)
    if path is None:
        raise RuntimeError("CUDA is unavailable, so there is no CUDA device to write tuning for")
    for settings in entries.values():
        check(settings)
    data, current = read_cache(device)
    merged = dict(data.get("entries", {}) if current else {})
    for key, settings in entries.items():
        merged[key] = {**merged.get(key, {}), **settings}
    where = _dev(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stamp": stamp(where), "entries": merged}, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    _files.pop(path, None)
    _bump()
    return path


_set: dict[str, int] = {}
_scopes: contextvars.ContextVar[tuple[Mapping[str, int], ...]] = contextvars.ContextVar("tuning_scopes", default=())
_derived: dict[tuple[str, str], int | None] = {}
_MISSING = object()


def lookup(name, *, path=None, size=None, batch=None, device=None, default=_MISSING, measured=True):
    """Return ``(value, source)`` for setting ``name``; the arguments are those of ``get``.

    ``source`` is ``"override"``, ``"set"``, ``"env"``, ``"cache"``, ``"derived"``, ``"default"`` or
    ``"fallback"``.

    Args:
        name: A key of ``SETTINGS``.
        path: Render path of the call.
        size: Frame size in pixels.
        batch: Number of environments in the call.
        device: Device the work runs on.
        default: The call site's value.
        measured: Read the tuning cache; False skips it.

    Raises:
        ValueError: If ``name`` is not a setting or its environment variable is invalid.
    """
    if name not in SETTINGS:
        check({name: None})
    setting = SETTINGS[name]
    for scope in reversed(_scopes.get()):
        if name in scope:
            return scope[name], "override"
    if name in _set:
        return _set[name], "set"
    value = _environ.read(setting.env_var)
    if value is not None:
        return value, "env"
    where = _where(setting, device)
    if where is not None:
        if measured:
            data, current = read_cache(where)
            key = DEVICE_ENTRY if path is None else entry_key(path, size or 0, batch or 1)
            value = data["entries"].get(key, {}).get(name) if current else None
            if value is not None:
                return value, "cache"
        if setting.derive is not None:
            if (name, where) not in _derived:
                _derived[name, where] = setting.derive(where)
            if _derived[name, where] is not None:
                return _derived[name, where], "derived"
    return (setting.fallback, "fallback") if default is _MISSING else (default, "default")


def sources_state(name: str, device=None) -> tuple:
    """Return a value that changes whenever a source of setting ``name`` on ``device`` may change.

    The sources are the ``override`` blocks of this thread, ``set``, the environment variable and the cache file,
    which ``read_cache`` reads again only after a bump of the setting generation. Callers keep a choice they made from
    the setting while this value stays equal. It calls no ``stat``.
    """
    setting = SETTINGS[name]
    where = _where(setting, device)
    return (_scopes.get(), _generation, os.environ.get(setting.env_var), None if where is None else cache_path(where))


def get(
    name: str,
    *,
    path: str | None = None,
    size: int | None = None,
    batch: int | None = None,
    device: Any = None,
    default: Any = _MISSING,
) -> int | None:
    """Return the current value of setting ``name``; ``lookup`` also returns where it came from.

    Args:
        name: A key of ``SETTINGS``.
        path: Render path of the call; with ``size`` and ``batch`` it selects measured values.
        size: Frame size in pixels (0 for simulation paths).
        batch: Number of environments in the call.
        device: Device the work runs on (the setting's own device if None).
        default: The call site's value, used instead of the setting's fallback.
    """
    return lookup(name, path=path, size=size, batch=batch, device=device, default=default)[0]


def set(**settings: int | None) -> dict[str, int]:  # noqa: A001 - mirrors ``snv.set``
    """Set settings for the process and return every setting set this way; None clears a setting.

    Values set here sit below ``override`` blocks and above everything else.

    Args:
        **settings: ``{setting name: value or None}``.
    """
    for name, value in check(settings).items():
        if value is None:
            _set.pop(name, None)
        else:
            _set[name] = value
    _bump()
    return dict(_set)


def reset() -> None:
    """Clear every setting set with ``set``."""
    _set.clear()
    _bump()


@contextlib.contextmanager
def override(**settings: int) -> Iterator[None]:
    """Set settings inside a ``with`` block, above ``set``.

    Blocks nest and the innermost wins. Other threads don't see the values, and None values are ignored.

    Args:
        **settings: ``{setting name: value}``.
    """
    values = {k: v for k, v in check(settings).items() if v is not None}
    token = _scopes.set((*_scopes.get(), values))
    _bump()
    try:
        yield
    finally:
        _scopes.reset(token)
        _bump()


def describe(**where) -> dict[str, tuple[int | None, str]]:
    """Return ``{name: (value, source)}`` for every setting.

    Args:
        **where: ``path``, ``size``, ``batch`` and ``device``, as for ``get``.
    """
    return {name: lookup(name, **where) for name in SETTINGS}


_noticed: dict[tuple[str, str], None] = {}


def on_render(path: str, size: int, batch: int, env) -> None:
    """Log a one-time ``stable-envs tune`` hint, or autotune, the first time a large untuned path renders.

    Runs once per process, path and device, with a batch of 256 or more. Never raises.

    Args:
        path: Render path that just ran.
        size: Frame size in pixels.
        batch: Number of rendered rows.
        env: The environment that rendered.
    """
    try:
        key = (path, str(env.device))
        if batch < HINT_MIN_ENVS or key in _noticed:
            return
        _noticed[key] = None
        settings = [k for k in SETTINGS.values() if path in k.paths and k.unit is not None]
        if not any(k.device in ("any", env.device.type) for k in settings):
            return
        data, current = read_cache(str(env.device))
        entries = data.get("entries", {}) if current else {}
        if entry_key(path, size, batch) in entries or any(k.name in entries.get(DEVICE_ENTRY, {}) for k in settings):
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
    """Tune a fresh ``name`` environment once, under a lock beside the cache file; keep the defaults on contention.

    Args:
        name: Registered class name.
        size: Frame size in pixels.
        batch: Number of environments.
        device: Device to tune.
    """
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
    _bump()
    _files.clear()
    _derived.clear()
    _noticed.clear()
    _cuda.cache_clear()
    stamp.cache_clear()
    _cache_file.cache_clear()
    _environ._parsed.cache_clear()


def render_paths(device="cuda"):
    """Return ``(path, renderers, measure)`` of the render paths that drew in this process on ``device``.

    Classes with a CPU and a GPU renderer register them when they draw; ``renderers`` has the ``small`` (CPU) and
    ``large`` renderer names, and ``time_backend`` times either one.

    Args:
        device: ``"cpu"`` or ``"cuda"``.
    """
    from ._execution.backends import registered

    return registered(device)


def time_backend(measure, backend: str, num_envs: int, device="cuda") -> float:
    """Seconds per draw call of renderer ``backend`` with ``num_envs`` environments, as ``stable-envs tune`` times it.

    Args:
        measure: The ``measure`` of a ``render_paths`` entry.
        backend: A renderer name of that entry.
        num_envs: Number of environments.
        device: Device of the outputs.
    """
    from ._execution.backends import time_backend as timed

    return timed(measure, backend, num_envs, device)
