"""Select the simulation backend or the renderer of a batch from the device, the batch size and a measured switch point.

A family declares, per device type, the backend it uses for small batches and the one it uses for large batches.
With ``simulation_backend="auto"``, a batch of ``large_backend_min_envs`` or more rows uses the large-batch backend.
Renderers switch the same way at ``large_renderer_min_rows`` rows (see ``select_renderer``). The switch point comes
from the knob, else from the tuning cache that ``stable-envs tune`` writes, else from the value the family ships in
its declaration. Nothing is timed when an environment is built or draws, so without tuning the same batch size
selects the same backend on every machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import torch

from .. import tuning
from .._core.config import log
from .._core.stepping import _threads

#: The knob that holds the switch point.
KNOB = "large_backend_min_envs"
#: The knob that holds the switch point of renderers.
RENDER_KNOB = "large_renderer_min_rows"
#: Request that selects a backend from the device and the batch size.
AUTO = "auto"

#: ``measure(backend, num_envs)`` builds ``backend`` with ``num_envs`` rows and returns ``(step, close)``:
#: ``step()`` advances (or, for a renderer, draws) every row once, and ``close()`` releases the backend.
Measure = Callable[[str, int], tuple[Callable[[], object], Callable[[], None]]]


def _seconds_per_step(step: Callable[[], object], device: torch.device) -> float:
    """Return the seconds of one ``step()`` call after two warm-up calls: the fastest mean of three windows of at
    least 0.05 s and 3 calls each, so that a burst of other work on the host doesn't decide the choice."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    # Steps keep torch on one thread while a native pool advances the batch; the timing does the same.
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
    """Seconds per step call of ``backend`` with ``num_envs`` rows, as ``stable-envs tune`` times it."""
    step, close = measure(backend, num_envs)
    try:
        return _seconds_per_step(step, torch.device(device))
    finally:
        close()


def measure_env(make: Callable[[str, int], object]) -> Measure:
    """A ``Measure`` for environments: ``make(backend, num_envs)`` builds one with that simulation backend.

    Each timed call is ``env.step`` without frames and with the State built, so the timing includes every cost the
    simulation backend adds per call, such as moving the State and the outputs. Rendering has its own switch and its
    own timing. The environment is reset with seed 0 and stepped with the same actions, drawn from its action spec,
    at every call.
    """

    def measure(backend: str, num_envs: int):
        import numpy as np

        env = make(backend, num_envs)
        env.reset(seed=0)
        actions = env.action_spec.sample(num_envs, np.random.default_rng(0), device=env.device)
        return (lambda: env.step(actions, frames=None)), env.close

    return measure


def measure_adapter(make: Callable[[str, int], object], actions: Callable[[int], object]) -> Measure:
    """A ``Measure`` for adapters: ``make(backend, num_envs)`` builds one, ``actions(num_envs)`` gives its actions.

    The adapter is reset with seed 0 and stepped with the same actions at every call.
    """

    def measure(backend: str, num_envs: int):
        adapter = make(backend, num_envs)
        adapter.reset(0)
        batch = actions(num_envs)
        return (lambda: adapter.step(batch)), adapter.close

    return measure


def _cached_neighbours(path: str, device, num_envs: int, knob: str = KNOB) -> str | None:
    """Return ``"large"`` or ``"small"`` when a cached range of ``path`` decides ``num_envs``.

    The small-batch backend steps rows with a cost that grows with the batch, and the large-batch backend has a
    per-call cost that is close to constant at small batches. So a smaller range that chose the large-batch backend
    also decides larger batches, and a larger range that chose the small-batch backend decides smaller ones.
    """
    data, current = tuning.read_cache(device)
    if not current:
        return None
    lo, _ = tuning.batch_range(num_envs)
    for key, knobs in data.get("entries", {}).items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] != path or knob not in knobs:
            continue
        a, b = (int(x) for x in parts[2].split("-"))
        large = knobs[knob] <= a
        if large and b < lo:
            return "large"
        if not large and a > num_envs:
            return "small"
    return None


def _calibrate(path: str, num_envs: int, device: torch.device, small: str, large: str, measure: Measure) -> bool:
    """Time both backends and return True when the large-batch backend is faster at ``num_envs`` rows.

    The small-batch backend is timed at 1, 2, 4, ... rows up to ``num_envs``. Its step time doesn't fall as the
    batch grows, so the search stops as soon as it is slower than the large-batch backend at ``num_envs``.
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


class Rungs(NamedTuple):
    """The backends of one device type and the shipped switch point between them."""

    small: str
    large: str
    #: Smallest batch that uses ``large`` when neither the knob nor the tuning cache sets the switch point. The
    #: family records next to the value how it was measured. None uses ``large`` at every batch size.
    min_envs: int | None = None


#: ``(knob, path, device type) -> (rungs, measure)`` of the simulation and render paths that this process selected
#: for; ``stable-envs tune`` calibrates them.
_REGISTERED: dict[tuple[str, str, str], tuple[Rungs, Measure]] = {}


def _rungs(backends: Mapping[str, Rungs], device: torch.device) -> Rungs:
    return Rungs(*backends.get(device.type, backends["cpu"]))


#: ``(knob, path, device, rows, rungs) -> (tuning state, rung)``: a choice holds while the knob's sources don't change.
_PICKS: dict[tuple, tuple[tuple, str]] = {}


def _pick(
    knob: str,
    path: str,
    num_envs: int,
    device: torch.device,
    rungs: Rungs,
    measure: Measure | None,
    use_knob: bool = True,
) -> str:
    """The rung of ``num_envs`` rows: from the knob, else the tuning cache, else the shipped switch point. With
    ``use_knob`` False, the knob is not read; the tuning cache entry of ``path`` and the switch point still apply."""
    if measure is not None:
        _REGISTERED[knob, path, device.type] = (rungs, measure)
    if rungs.small == rungs.large:
        return rungs.small
    key = (knob, path, str(device), num_envs, rungs, use_knob)
    state = tuning.sources_state(knob, device)
    hit = _PICKS.get(key)
    if hit is not None and hit[0] == state:
        return hit[1]
    rung = _choose(knob, path, num_envs, device, rungs, use_knob)
    _PICKS[key] = (state, rung)
    return rung


def _choose(knob: str, path: str, num_envs: int, device: torch.device, rungs: Rungs, use_knob: bool = True) -> str:
    if use_knob:
        value = tuning.get(knob, path=path, size=0, batch=num_envs, device=device)
    else:
        data, current = tuning.read_cache(device)
        entry = data.get("entries", {}).get(tuning.entry_key(path, 0, num_envs), {}) if current else {}
        value = entry.get(knob)
    if value is not None:
        return rungs.large if num_envs >= value else rungs.small
    side = _cached_neighbours(path, device, num_envs, knob)
    if side is not None:
        return rungs.large if side == "large" else rungs.small
    return rungs.large if rungs.min_envs is None or num_envs >= rungs.min_envs else rungs.small


def select_backend(
    path: str,
    num_envs: int,
    device: str | torch.device,
    requested: str | None,
    backends: Mapping[str, Rungs],
    *,
    choices: Sequence[str],
    measure: Measure | None = None,
    knob: bool = True,
) -> str:
    """Return the simulation backend of a batch. It reads the knob and the tuning cache, and times nothing.

    Args:
        path: Simulation path of the family and task, for example ``"dmc-sim/reacher"``. It keys the cache.
        num_envs: Number of rows in the batch.
        device: Device of the outputs.
        requested: ``"auto"`` (or None) to select from ``backends``, else one of ``choices``.
        backends: ``{device type: Rungs}``. A device type that isn't a key uses the ``"cpu"`` entry. This is the
            family's one declaration of its backends and shipped switch points.
        choices: Backends that ``requested`` may name.
        measure: Builds and steps one backend; ``stable-envs tune`` uses it to calibrate ``path``.
        knob: False ignores the ``large_backend_min_envs`` knob for this path, for a split that the knob doesn't
            describe. The tuning cache and the shipped switch point still apply.

    Raises:
        ValueError: If ``requested`` is neither ``"auto"``, None nor one of ``choices``.
    """
    if requested is None:
        requested = AUTO
    if requested != AUTO:
        if requested not in choices:
            names = ", ".join(repr(c) for c in (AUTO, *choices))
            raise ValueError(f"simulation_backend must be one of {names}; got {requested!r}")
        return requested
    device = torch.device(device)
    return _pick(KNOB, path, num_envs, device, _rungs(backends, device), measure, use_knob=knob)


def select_renderer(
    path: str,
    rows: int,
    device: str | torch.device,
    renderers: Mapping[str, Rungs],
    *,
    measure: Measure | None = None,
) -> str:
    """Return the renderer of a batch of ``rows`` frames. It reads the knob and the tuning cache, and times nothing.

    The rule is that of ``select_backend``, with the ``large_renderer_min_rows`` knob.

    Args:
        path: Render path of the class, for example ``"navigation/TwoRoom"``. It keys the cache.
        rows: Number of frames in the call.
        device: Device of the frames.
        renderers: ``{device type: Rungs}``. A device type that isn't a key uses the ``"cpu"`` entry.
        measure: Builds and draws with one renderer; ``stable-envs tune`` uses it to calibrate ``path``.
    """
    device = torch.device(device)
    return _pick(RENDER_KNOB, path, rows, device, _rungs(renderers, device), measure)


def registered(
    device: str | torch.device, prefixes: Sequence[str] = (), paths: Sequence[str] = (), knob: str = KNOB
) -> list[tuple[str, Rungs, Measure]]:
    """``(path, rungs, measure)`` of the paths selected for with ``knob`` on the type of ``device``; with ``paths``,
    only those paths; with ``prefixes``, only the paths whose first part is one of them."""
    kind = torch.device(device).type
    return [
        (path, rungs, measure)
        for (name, path, where), (rungs, measure) in _REGISTERED.items()
        if name == knob
        and where == kind
        and (not paths or path in paths)
        and (not prefixes or path.split("/")[0] in prefixes)
    ]


def calibrate(
    num_envs: int,
    device: str | torch.device,
    *,
    prefixes: Sequence[str] = (),
    paths: Sequence[str] = (),
    write: bool = True,
    knob: str = KNOB,
) -> dict[str, str]:
    """Time both backends of the paths registered with ``knob`` on ``device`` (see ``registered``) at ``num_envs``
    rows.

    ``stable-envs tune`` calls it for the environment it tunes, once per knob. Each choice goes to the tuning cache of
    the device unless ``write`` is False. Returns ``{path: chosen backend}``.
    """
    device = torch.device(device)
    chosen = {}
    for path, rungs, measure in registered(device, prefixes, paths, knob):
        if rungs.small == rungs.large:
            continue
        wins = _calibrate(path, num_envs, device, rungs.small, rungs.large, measure)
        chosen[path] = rungs.large if wins else rungs.small
        if write:
            lo, hi = tuning.batch_range(num_envs)
            tuning.write_cache({tuning.entry_key(path, 0, num_envs): {knob: lo if wins else hi + 1}}, device)
    return chosen
