"""``stable-envs tune``: time the knobs of one render path on this machine and cache the fastest values."""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from . import tuning

STEPS = 4
REPEATS = 5
#: Rough seconds per timed call and per environment build, for the ``--all`` estimate.
_CALL_SECONDS, _BUILD_SECONDS = 0.25, 5.0


def knobs_for(env) -> list[tuning.Knob]:
    """Return the tunable knobs ``env`` reads on its device."""
    paths = set(tuning.paths_of(type(env)))
    kind = env.device.type
    out = []
    for k in tuning.KNOBS.values():
        if not k.tunable or not paths & set(k.paths) or k.device not in ("any", kind):
            continue
        if k.workload == "step" and not env.has_dynamics:
            continue
        out.append(k)
    return out


def _cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _effective(knob: tuning.Knob, value: int, env) -> int:
    """Map a value to the work split it produces; values with equal splits time the same."""
    batch = env.num_envs
    if knob.unit == "rows":
        return min(value, batch)
    if knob.unit == "rays":
        return min(batch, max(1, value // (env.size * env.size * env.ss * env.ss)))
    if knob.unit == "cpus":
        return min(value, _cpus(), batch)
    return value


def _fits(knob: tuning.Knob, value: int, env) -> bool:
    """Whether ``value`` keeps the knob's memory within half the free GPU memory."""
    if env.device.type != "cuda":
        return True
    free = tuning.free_cuda_bytes(env.device)
    if free is None:
        return True
    if knob.unit == "bytes":
        return value <= free // 2
    if knob.name == "dmc_ray_budget":
        return value * tuning.DMC_BYTES_PER_RAY <= free // 2
    return True


def default_value(knob: tuning.Knob, env) -> int | None:
    """The value ``knob`` takes for ``env`` with nothing set and nothing measured."""
    defaults = getattr(env, "_tuning_defaults", None)
    default = defaults(env.size).get(knob.name) if defaults else None
    with tuning.running(env):
        value, source = tuning._resolve(knob.name, env.device, measured=False)
    return default if default is not None and source == "fallback" else value


def candidates(knob: tuning.Knob, start: int, env) -> list[int]:
    """Values to time: ``start`` and its powers-of-two neighbours, pruned to distinct work splits that fit."""
    if knob.choices:
        raw = [c for c in knob.choices if start / 4 <= c <= start * 4]
    else:
        raw = [start >> 2, start >> 1, start, start << 1, start << 2]
        if knob.unit == "cpus":
            raw = [0, *raw]
    seen, out = {}, []
    for value in sorted({v for v in raw if v >= knob.minimum}):
        try:
            knob.check(value)
        except ValueError:
            continue
        if value != start and not _fits(knob, value, env):
            continue
        split = _effective(knob, value, env)
        if split in seen and value != start:
            continue
        if split in seen:
            out.remove(seen[split])
        seen[split] = value
        out.append(value)
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _same(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    fa, fb = (dict(x.flatten_keys(".").items()) for x in (a, b))
    return fa.keys() == fb.keys() and all(torch.equal(fa[k].cpu(), fb[k].cpu()) for k in fa)


def _time(env, knob: tuning.Knob, repeats: int) -> tuple[float, object]:
    try:
        state = env.reset(seed=0)
        render = env.render
    except NotImplementedError:
        # Render-only environments draw sampled renderer rows.
        state = torch.as_tensor(env.sample_states(env.num_envs, np.random.default_rng(0)), device=env.device)
        render = env._render
    if knob.workload == "step":
        rng = np.random.default_rng(0)
        actions = torch.stack([env.action_spec.sample(env.num_envs, rng=rng) for _ in range(STEPS)], 1)

        def work():
            env.set_state(state)
            return env.step(actions, states=None, frames=None)["state"]

    else:

        def work():
            return render(state)

    result = work()
    _sync(env.device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        work()
        _sync(env.device)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def tune(
    env, *, knobs: Sequence[str] = (), repeats: int = REPEATS, write: bool = True, echo: Callable | None = print
) -> dict[str, int]:
    """Time the knobs ``env`` reads, one at a time from the default, and cache the fastest values.

    Values are saved for the environment's render path, resolution and batch range on this device,
    so they apply to every environment on the same path. The environment's State is restored afterwards.
    """
    from . import registry

    try:
        saved = env.get_state()
    except (NotImplementedError, RuntimeError):
        saved = None
    try:
        return _sweep(env, knobs, repeats, write, echo, registry)
    finally:
        if saved is not None:
            env.set_state(saved)


def _sweep(env, knobs, repeats, write, echo, registry) -> dict[str, int]:
    say = echo or (lambda *_: None)
    chosen = knobs_for(env)
    if knobs:
        tuning.check({k: None for k in knobs})
        chosen = [k for k in chosen if k.name in knobs]
    if not chosen:
        say(f"{env.class_name} reads no tunable knob on {env.device.type}.")
        return {}
    say(
        f"tuning {env.class_name} on {tuning.device_name(env.device)}: "
        f"{env.size} px, batch {env.num_envs}, {len(chosen)} knob(s)"
    )
    kwargs = dict(num_envs=env.num_envs, size=env.size, device=str(env.device))
    best: dict[str, int] = {}
    rows = []
    for knob in chosen:
        start = default_value(knob, env)
        if start is None:
            continue
        times, reference = {}, None
        for value in candidates(knob, start, env):
            with tuning.override(**best, **{knob.name: value}):
                if knob.rebuild:
                    with registry.make(env.class_name, **kwargs) as fresh:
                        seconds, result = _time(fresh, knob, repeats)
                else:
                    seconds, result = _time(env, knob, repeats)
            if reference is None:
                reference = result
            elif not _same(reference, result):
                raise RuntimeError(
                    f"{knob.name}={value} changed the {knob.workload} output of {env.class_name}; tuning knobs "
                    "must only change speed. Please report this as a bug."
                )
            times[value] = seconds
        pick = min(times, key=lambda v: (times[v], v != start))
        best[knob.name] = pick
        rows.append((knob.name, start, pick, times[start] / times[pick]))
    say(f"  {'knob':<22} {'default':>12} {'tuned':>12} {'speedup':>8}")
    for name, start, pick, gain in rows:
        say(f"  {name:<22} {start:>12} {pick:>12} {gain:>7.2f}x")
    if write and best:
        _save(env, best, say)
    return best


def _save(env, best: dict[str, int], say: Callable) -> None:
    from . import registry

    paths = tuning.paths_of(type(env))
    files: dict[str, dict] = {}
    for name, value in best.items():
        knob = tuning.KNOBS[name]
        path = next(p for p in paths if p in knob.paths)
        device = "cpu" if knob.device == "cpu" else str(env.device)
        key = tuning.entry_key(path, env.size, env.num_envs)
        files.setdefault(device, {}).setdefault(key, {"knobs": {}})["knobs"][name] = value
    for device, entries in files.items():
        tuning.write_cache(entries, device)
    lo, hi = tuning.batch_range(env.num_envs)
    for path in sorted({k.split("|")[0] for e in files.values() for k in e}):
        others = [
            n for n in registry.list_envs() if n != env.class_name and path in tuning.paths_of(registry.REGISTRY[n])
        ]
        also = ", ".join(others[:5]) + (f" and {len(others) - 5} more" if len(others) > 5 else "")
        say(f"saved for: {path}, {env.size} px, batch {lo}-{hi}" + (f" (also covers: {also})" if also else ""))
    for device in files:
        say(f"cache: {tuning.cache_path(device)}")


def representatives(device: str) -> dict[str, str]:
    """Return one registered environment per tunable path on ``device``."""
    from . import registry

    kind = torch.device(device).type
    wanted = {p for k in tuning.KNOBS.values() if k.tunable and k.device in ("any", kind) for p in k.paths}
    out: dict[str, str] = {}
    for name in registry.list_envs():
        cls = registry.REGISTRY[name]
        if kind == "cpu" and getattr(cls, "requires_cuda", False):
            continue
        for path in tuning.paths_of(cls):
            if path in wanted and path not in out:
                out[path] = name
    return out


def estimate_seconds(names: Sequence[str], device: str, repeats: int = REPEATS) -> float:
    """Rough wall time of tuning ``names``: about five values per knob, each timed ``repeats + 1`` times."""
    from . import registry

    kind = torch.device(device).type
    total = 0.0
    for name in names:
        cls = registry.REGISTRY[name]
        paths = set(tuning.paths_of(cls))
        for k in tuning.KNOBS.values():
            if k.tunable and paths & set(k.paths) and k.device in ("any", kind):
                total += 5 * ((repeats + 1) * _CALL_SECONDS + (_BUILD_SECONDS if k.rebuild else 0.0))
        total += _BUILD_SECONDS
    return total


def show(echo: Callable = print) -> int:
    """Print the cached entries of this machine's devices."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    found = False
    for device in devices:
        data, current = tuning.read_cache(device)
        if not data:
            continue
        found = True
        stamp = data.get("stamp", {})
        state = "current" if current else "stale (device, driver or version changed; ignored)"
        echo(f"{tuning.cache_path(device)}: {stamp.get('device')}, stable-envs {stamp.get('stable_envs')}, {state}")
        for key, entry in sorted(data.get("entries", {}).items()):
            path, size, batch = key.split("|")
            values = ", ".join(f"{k}={v}" for k, v in sorted(entry.get("knobs", {}).items()))
            echo(f"  {path:<18} {size:>5} px  batch {batch:<10} {values}")
    if not found:
        echo("no tuning cached on this machine; every knob uses its derived default")
    return 0


def clear(echo: Callable = print) -> int:
    """Delete the cache files of this machine's devices."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        path = tuning.cache_path(device)
        if path is not None and path.exists():
            path.unlink()
            echo(f"deleted {path}")
    tuning._clear_caches()
    return 0


def status(device: str) -> str:
    """One line on how many tunable paths have measured values on ``device``."""
    wanted = representatives(device)
    data, current = tuning.read_cache(device)
    tuned = {k.split("|")[0] for k in data.get("entries", {})} if current else set()
    done = len(tuned & set(wanted))
    return f"{done} of {len(wanted)} render paths tuned on {tuning.device_name(device)}; others use derived defaults"


def main(args) -> int:
    """Run the ``tune`` subcommand from parsed arguments."""
    from . import registry
    from .env import Env

    if args.show:
        return show()
    if args.clear:
        return clear()
    device = Env.resolve_device(args.device)
    if args.all:
        reps = representatives(device)
        minutes = estimate_seconds(list(reps.values()), device, args.repeats) / 60
        print(f"tuning {len(reps)} render paths on {tuning.device_name(device)}, about {minutes:.0f} min:")
        for path, name in reps.items():
            print(f"  {path:<18} {name}")
        if not args.yes:
            if not sys.stdin.isatty():
                print("pass --yes to start without a prompt")
                return 1
            if input("start? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        names = list(reps.values())
    elif args.env:
        names = [args.env]
    else:
        print("name an environment, or pass --all, --show or --clear")
        return 2
    for name in dict.fromkeys(names):
        kwargs = dict(num_envs=args.batch, device=device)
        if args.size is not None:
            kwargs["size"] = args.size
        with registry.make(name, **kwargs) as env:
            tune(env, knobs=args.knob or (), repeats=args.repeats)
    return 0
