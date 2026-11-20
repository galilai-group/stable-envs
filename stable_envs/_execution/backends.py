"""Select the simulation backend of a batch, and the renderer of a batch from a measured switch point.

A family declares one simulation backend per device type. ``simulation_backend="auto"`` selects it at every batch
size, so a seed gives the same trajectory at every batch size. Other backends are selected by name.

A class with a CPU and a CUDA renderer that draw the same bytes declares, per device type, the renderer it uses for
small batches and the one it uses for large batches. A call of ``large_renderer_min_rows`` or more rows uses the
large-batch renderer (see ``select_renderer``). The switch point comes from the setting, else from the tuning cache
that ``stable-envs tune`` writes, else from the value the class ships. Nothing is timed when an environment draws, so
without tuning the same batch size selects the same renderer on every machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import torch

from .. import tuning
from .._core.config import log
from .._core.stepping import _threads

#: The setting that holds the switch point of renderers.
RENDERER_SETTING = "large_renderer_min_rows"
#: Request that selects the default backend of the device.
AUTO = "auto"

#: ``measure(renderer, rows)`` builds an environment with ``rows`` rows and returns ``(draw, close)``: ``draw()`` draws
#: every row once with ``renderer``, and ``close()`` releases the environment.
Measure = Callable[[str, int], tuple[Callable[[], object], Callable[[], None]]]


def _seconds_per_step(step: Callable[[], object], device: torch.device) -> float:
    """Return the seconds of one ``step()`` call after two warm-up calls: the fastest mean of three windows of at
    least 0.05 s and 3 calls each, so that a burst of other work on the host doesn't decide the choice."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    # The CPU renderers keep torch on one thread while a native pool draws; the timing does the same.
    with _threads(1):
        for _ in range(2):
            step()
        sync()
        best = float("inf")
        for _ in range(3):
            calls, start = 0, time.perf_counter()
            while calls < 3 or time.perf_counter() - start < 0.05:
                step()
                calls += 1
                if calls % 5 == 0:
                    sync()
            sync()
            best = min(best, (time.perf_counter() - start) / calls)
        return best


def time_backend(measure: Measure, backend: str, num_envs: int, device: str | torch.device) -> float:
    """Seconds per draw call of renderer ``backend`` with ``num_envs`` rows, as ``stable-envs tune`` times it."""
    step, close = measure(backend, num_envs)
    try:
        return _seconds_per_step(step, torch.device(device))
    finally:
        close()


def _cached_neighbours(path: str, device, num_envs: int) -> str | None:
    """Return ``"large"`` or ``"small"`` when a cached range of ``path`` decides ``num_envs``.

    The CPU renderer draws rows with a cost that grows with the batch, and the CUDA renderer has a per-call cost that
    is close to constant at small batches. So a smaller range that chose the CUDA renderer also decides larger
    batches, and a larger range that chose the CPU renderer decides smaller ones.
    """
    data, current = tuning.read_cache(device)
    if not current:
        return None
    lo, _ = tuning.batch_range(num_envs)
    for key, settings in data.get("entries", {}).items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] != path or RENDERER_SETTING not in settings:
            continue
        a, b = (int(x) for x in parts[2].split("-"))
        large = settings[RENDERER_SETTING] <= a
        if large and b < lo:
            return "large"
        if not large and a > num_envs:
            return "small"
    return None


def _calibrate(path: str, num_envs: int, device: torch.device, small: str, large: str, measure: Measure) -> bool:
    """Time both renderers and return True when the large-batch renderer is faster at ``num_envs`` rows.

    The small-batch renderer is timed at 1, 2, 4, ... rows up to ``num_envs``. Its draw time doesn't fall as the
    batch grows, so the search stops as soon as it is slower than the large-batch renderer at ``num_envs``.
    """
    start = time.perf_counter()
    try:
        large_s = time_backend(measure, large, num_envs, device)
    except Exception as e:  # noqa: BLE001 - a backend that can't run loses
        log("WARNING", f"{path}: timing the {large} backend failed ({type(e).__name__}: {e}); using {small}.")
        return False
    rows, small_s = 1, 0.0
    while True:
        rows = min(rows, num_envs)
        small_s = time_backend(measure, small, rows, device)
        if small_s > large_s or rows == num_envs:
            break
        rows *= 2
    wins = small_s > large_s
    log(
        "INFO",
        f"{path}: {large} takes {large_s * 1e6:.0f} us for {num_envs} rows, {small} takes {small_s * 1e6:.0f} us "
        f"for {rows} rows; using {large if wins else small} "
        f"(measured in {time.perf_counter() - start:.1f} s, cached).",
    )
    return wins


class BackendPair(NamedTuple):
    """The renderers of one device type and the shipped switch point between them."""

    small: str
    large: str
    #: Smallest batch that uses ``large`` when neither the setting nor the tuning cache sets the switch point. The
    #: family records next to the value how it was measured. None uses ``large`` at every batch size.
    min_envs: int | None = None


#: ``(path, device type) -> (pair, measure)`` of the render paths that this process selected for; ``stable-envs tune``
#: calibrates them.
_REGISTERED: dict[tuple[str, str], tuple[BackendPair, Measure]] = {}


def _pair_of(backends: Mapping[str, BackendPair], device: torch.device) -> BackendPair:
    return BackendPair(*backends.get(device.type, backends["cpu"]))


#: ``(path, device, rows, pair) -> (tuning state, renderer)``: a choice holds while the setting's sources don't change.
_PICKS: dict[tuple, tuple[tuple, str]] = {}


def _pick(path: str, num_envs: int, device: torch.device, pair: BackendPair, measure: Measure | None) -> str:
    """The renderer of ``num_envs`` rows: from the setting, else the tuning cache, else the shipped switch point."""
    if measure is not None:
        _REGISTERED[path, device.type] = (pair, measure)
    if pair.small == pair.large:
        return pair.small
    key = (path, str(device), num_envs, pair)
    state = tuning.sources_state(RENDERER_SETTING, device)
    hit = _PICKS.get(key)
    if hit is not None and hit[0] == state:
        return hit[1]
    backend = _choose(path, num_envs, device, pair)
    _PICKS[key] = (state, backend)
    return backend


def _choose(path: str, num_envs: int, device: torch.device, pair: BackendPair) -> str:
    value = tuning.get(RENDERER_SETTING, path=path, size=0, batch=num_envs, device=device)
    if value is not None:
        return pair.large if num_envs >= value else pair.small
    side = _cached_neighbours(path, device, num_envs)
    if side is not None:
        return pair.large if side == "large" else pair.small
    return pair.large if pair.min_envs is None or num_envs >= pair.min_envs else pair.small


def select_backend(
    device: str | torch.device, requested: str | None, defaults: Mapping[str, str], *, choices: Sequence[str]
) -> str:
    """Return the simulation backend of a batch.

    Args:
        device: Device of the outputs.
        requested: ``"auto"`` (or None) to select the default of the device type, else one of ``choices``.
        defaults: ``{device type: backend}``. A device type that isn't a key uses the ``"cpu"`` entry. This is the
            family's one declaration of its default backends.
        choices: Backends that ``requested`` may name.

    Raises:
        ValueError: If ``requested`` is neither ``"auto"``, None nor one of ``choices``.
    """
    if requested is None or requested == AUTO:
        kind = torch.device(device).type
        return defaults.get(kind, defaults["cpu"])
    if requested not in choices:
        names = ", ".join(repr(c) for c in (AUTO, *choices))
        raise ValueError(f"simulation_backend must be one of {names}; got {requested!r}")
    return requested


def select_renderer(
    path: str,
    rows: int,
    device: str | torch.device,
    renderers: Mapping[str, BackendPair],
    *,
    measure: Measure | None = None,
) -> str:
    """Return the renderer of a batch of ``rows`` frames. It reads the setting and the tuning cache, and times nothing.

    Args:
        path: Render path of the class, for example ``"navigation/TwoRoom"``. It keys the cache.
        rows: Number of frames in the call.
        device: Device of the frames.
        renderers: ``{device type: BackendPair}``. A device type that isn't a key uses the ``"cpu"`` entry.
        measure: Builds and draws with one renderer; ``stable-envs tune`` uses it to calibrate ``path``.
    """
    device = torch.device(device)
    return _pick(path, rows, device, _pair_of(renderers, device), measure)


def registered(
    device: str | torch.device, prefixes: Sequence[str] = (), paths: Sequence[str] = ()
) -> list[tuple[str, BackendPair, Measure]]:
    """``(path, pair, measure)`` of the render paths selected for on the type of ``device``; with ``paths``, only those
    paths; with ``prefixes``, only the paths whose first part is one of them."""
    kind = torch.device(device).type
    return [
        (path, pair, measure)
        for (path, where), (pair, measure) in _REGISTERED.items()
        if where == kind and (not paths or path in paths) and (not prefixes or path.split("/")[0] in prefixes)
    ]


def calibrate(
    num_envs: int,
    device: str | torch.device,
    *,
    prefixes: Sequence[str] = (),
    paths: Sequence[str] = (),
    write: bool = True,
) -> dict[str, str]:
    """Time both renderers of the paths registered on ``device`` (see ``registered``) at ``num_envs`` rows.

    ``stable-envs tune`` calls it for the environment it tunes. Each choice goes to the tuning cache of the device
    unless ``write`` is False. Returns ``{path: chosen renderer}``.
    """
    device = torch.device(device)
    chosen = {}
    for path, pair, measure in registered(device, prefixes, paths):
        if pair.small == pair.large:
            continue
        wins = _calibrate(path, num_envs, device, pair.small, pair.large, measure)
        chosen[path] = pair.large if wins else pair.small
        if write:
            lo, hi = tuning.batch_range(num_envs)
            tuning.write_cache(
                {tuning.entry_key(path, 0, num_envs): {RENDERER_SETTING: lo if wins else hi + 1}}, device
            )
    return chosen
