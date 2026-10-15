"""``stable-envs tune``: time the knobs of one render path on this machine and cache the fastest values."""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np
import torch

from . import tuning

STEPS = 4


def knobs_for(env) -> list[tuning.Knob]:
    """Return the knobs ``tune`` sweeps for ``env`` on its device."""
    paths = tuning.paths_of(type(env))
    return [
        k
        for k in tuning.KNOBS.values()
        if k.unit is not None
        and any(p in k.paths for p in paths)
        and k.device in ("any", env.device.type)
        and (env.has_dynamics or not k.step)
    ]


def candidates(knob: tuning.Knob, start: int, env) -> list[int]:
    """Return ``start`` and its powers-of-two neighbours, one per distinct work split, within memory."""
    raw = [start >> 2, start >> 1, start, start << 1, start << 2] + ([0] if knob.unit == "cpus" else [])
    free = tuning.free_cuda_bytes(env.device) if env.device.type == "cuda" else None
    frame = env.size * env.size * env.ss * env.ss
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    split = {
        "rows": lambda v: min(v, env.num_envs),
        "rays": lambda v: min(env.num_envs, max(1, v // frame)),
        "cpus": lambda v: min(v, cpus, env.num_envs),
    }.get(knob.unit, lambda v: v)
    out: dict[int, int] = {}
    for value in sorted(raw):
        try:
            knob.check(value)
        except ValueError:
            continue
        rays = value * 256 if knob.name == "dmc_ray_budget" else value
        if value != start and free is not None and knob.unit in ("bytes", "rays") and rays > free // 2:
            continue
        if split(value) not in out or value == start:
            out[split(value)] = value
    return sorted(out.values())


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _same(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    fa, fb = dict(a.flatten_keys(".").items()), dict(b.flatten_keys(".").items())
    return fa.keys() == fb.keys() and all(torch.equal(fa[k].cpu(), fb[k].cpu()) for k in fa)


def _time(env, knob: tuning.Knob, repeats: int):
    if knob.step:
        state = env.reset(seed=0)
        rng = np.random.default_rng(0)
        actions = torch.stack([env.action_spec.sample(env.num_envs, rng=rng) for _ in range(STEPS)], 1)

        def work():
            env.set_state(state)
            return env.step(actions, states=None, frames=None)["state"]

    elif env.has_dynamics:
        state = env.reset(seed=0)

        def work():
            return env.render(state)

    else:
        rows = torch.as_tensor(env.sample_states(env.num_envs, np.random.default_rng(0)), device=env.device)

        def work():
            return env._render(rows)

    result = work()
    _sync(env.device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        work()
        _sync(env.device)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def tune(env, *, knobs=(), repeats: int = 5, write: bool = True, echo=print) -> dict[str, int]:
    """Sweep the knobs ``env`` reads one at a time from their defaults; cache and return the fastest values."""
    from . import registry

    say = echo or (lambda *_: None)
    tuning.check(dict.fromkeys(knobs))
    chosen = [k for k in knobs_for(env) if not knobs or k.name in knobs]
    if not chosen:
        say(f"{type(env).__name__} reads no tunable knob on {env.device.type}.")
        return {}
    paths = tuning.paths_of(type(env))
    try:
        saved = env.get_state() if env.has_dynamics else None
    except NotImplementedError:
        saved = None
    kwargs = dict(num_envs=env.num_envs, size=env.size, device=str(env.device))
    best, rows = {}, []
    try:
        for knob in chosen:
            path = next(p for p in paths if p in knob.paths)
            size = 0 if knob.step else env.size
            where = dict(path=path if knob.per_path else None, size=size, batch=env.num_envs, device=env.device)
            start = tuning.lookup(knob.name, measured=False, **where)[0]
            times, reference = {}, None
            for value in candidates(knob, start, env):
                with tuning.override(**best, **{knob.name: value}):
                    if knob.step:
                        with registry.make(type(env).__name__, **kwargs) as fresh:
                            seconds, result = _time(fresh, knob, repeats)
                    else:
                        seconds, result = _time(env, knob, repeats)
                if reference is not None and not _same(reference, result):
                    raise RuntimeError(
                        f"{knob.name}={value} changed the output of {type(env).__name__}; please report it"
                    )
                reference = result if reference is None else reference
                times[value] = seconds
            best[knob.name] = min(times, key=lambda v: (times[v], v != start))
            rows.append((knob, where, start, best[knob.name], times[start] / times[best[knob.name]]))
    finally:
        if saved is not None:
            env.set_state(saved)
    say(f"  {'knob':<22} {'default':>12} {'tuned':>12} {'speedup':>8}")
    for knob, _, start, pick, gain in rows:
        say(f"  {knob.name:<22} {start:>12} {pick:>12} {gain:>7.2f}x")
    if write:
        _save(env, rows, say)
    return best


def _save(env, rows, say) -> None:
    from . import registry

    files: dict[str, dict[str, dict[str, int]]] = {}
    for knob, where, _, pick, _ in rows:
        device = "cpu" if knob.device == "cpu" else str(env.device)
        key = (
            tuning.DEVICE_ENTRY
            if where["path"] is None
            else tuning.entry_key(where["path"], where["size"], env.num_envs)
        )
        files.setdefault(device, {}).setdefault(key, {})[knob.name] = pick
    for device, entries in files.items():
        path = tuning.write_cache(entries, device)
        for key in entries:
            if key == tuning.DEVICE_ENTRY:
                say(f"saved for: every render path on this device ({path})")
                continue
            name, size, batch = key.split("|")
            others = [n for n in registry.list_envs() if name in tuning.paths_of(registry.REGISTRY[n])]
            others = [n for n in others if n != type(env).__name__]
            also = ", ".join(others[:5]) + (f" and {len(others) - 5} more" if len(others) > 5 else "")
            say(
                f"saved for: {name}, {size} px, batch {batch}"
                + (f" (also covers: {also})" if also else "")
                + f" ({path})"
            )


def representatives(device: str) -> dict[str, str]:
    """Return one registered environment per tunable render path on ``device``."""
    from . import registry

    kind = torch.device(device).type
    wanted = {p for k in tuning.KNOBS.values() if k.unit is not None and k.device in ("any", kind) for p in k.paths}
    out: dict[str, str] = {}
    for name in registry.list_envs():
        cls = registry.REGISTRY[name]
        if kind == "cpu" and getattr(cls, "requires_cuda", False):
            continue
        for path in tuning.paths_of(cls):
            if path in wanted:
                out.setdefault(path, name)
    return out


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def show() -> int:
    """Print the cached values of this machine's devices."""
    found = False
    for device in _devices():
        data, current = tuning.read_cache(device)
        if data:
            found = True
            state = "current" if current else "stale (device, driver, CPU or version changed; ignored)"
            print(f"{tuning.cache_path(device)}: {data.get('stamp', {}).get('device')}, {state}")
            for key, knobs in sorted(data.get("entries", {}).items()):
                print(f"  {key:<36} " + ", ".join(f"{k}={v}" for k, v in sorted(knobs.items())))
    if not found:
        print("no tuning cached on this machine; every knob uses its derived default")
    return 0


def clear() -> int:
    """Delete the cache files of this machine's devices."""
    for device in _devices():
        path = tuning.cache_path(device)
        if path is not None and path.exists():
            path.unlink()
            print(f"deleted {path}")
    tuning._clear_caches()
    return 0


def status(device: str) -> str:
    """One line on how many tunable paths have measured values on ``device``."""
    wanted = representatives(device)
    data, current = tuning.read_cache(device)
    tuned = {k.split("|")[0] for k in data.get("entries", {})} if current else set()
    done = len(tuned & set(wanted))
    return f"{done} of {len(wanted)} render paths tuned on this {device} device; others use derived defaults"


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
        names = list(dict.fromkeys(representatives(device).values()))
        print(f"tuning {len(names)} environments on {device}, about {len(names)} min: {', '.join(names)}")
        if not args.yes and (not sys.stdin.isatty() or input("start? [y/N] ").strip().lower() not in ("y", "yes")):
            print("not started; pass --yes to start without a prompt")
            return 1
    elif args.env:
        names = [args.env]
    else:
        print("name an environment, or pass --all, --show or --clear")
        return 2
    for name in names:
        kwargs = dict(num_envs=args.batch, device=device) | ({"size": args.size} if args.size else {})
        with registry.make(name, **kwargs) as env:
            tune(env, knobs=args.knob or (), repeats=args.repeats)
    return 0
