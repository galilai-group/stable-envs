"""``stable-envs tune``: time the settings of one render path on this machine and cache the fastest values."""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

from . import tuning
from ._core.cpus import cpu_limit

STEPS = 4


def settings_for(env) -> list[tuning.Setting]:
    """Return the settings ``tune`` sweeps for ``env`` on its device."""
    paths = tuning.paths_of(type(env))
    return [
        k
        for k in tuning.SETTINGS.values()
        if k.unit is not None
        and any(p in k.paths for p in paths)
        and k.device in ("any", env.device.type)
        and (env.has_dynamics or not k.step)
    ]


def candidates(setting: tuning.Setting, start: int, env) -> list[int]:
    """Return ``start`` and its powers-of-two neighbors, one per distinct work split, within memory."""
    raw = [start >> 2, start >> 1, start, start << 1, start << 2] + ([0] if setting.unit == "cpus" else [])
    free = tuning.free_cuda_bytes(env.device) if env.device.type == "cuda" else None
    frame = env.size * env.size * env.ss * env.ss
    cpus = cpu_limit()
    split = {
        "rows": lambda v: min(v, env.num_envs),
        "rays": lambda v: min(env.num_envs, max(1, v // frame)),
        "cpus": lambda v: min(v, cpus, env.num_envs),
    }.get(setting.unit, lambda v: v)
    out: dict[int, int] = {}
    for value in sorted(raw):
        try:
            setting.check(value)
        except ValueError:
            continue
        rays = value * 256 if setting.name == "dmc_ray_budget" else value
        if value != start and free is not None and setting.unit in ("bytes", "rays") and rays > free // 2:
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


def _time(env, setting: tuning.Setting, repeats: int):
    if setting.step:
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


def tune(env, *, settings=(), repeats: int = 5, write: bool = True, echo=print) -> dict[str, int]:
    """Sweep the settings ``env`` reads one at a time from their defaults; cache and return the fastest values."""
    from . import registry
    from ._execution import backends

    say = echo or (lambda *_: None)
    tuning.check(dict.fromkeys(settings))
    if not settings or backends.BACKEND_SETTING in settings:
        # The simulation backends of the paths this env selected for: the one place that times them.
        # The env's own path when it names one: other environments built in this process keep their choices.
        own = getattr(env, "simulation_backend_path", None)
        scope = {"paths": (own,)} if own else {"prefixes": tuning.paths_of(type(env))}
        for path, pick in backends.calibrate(env.num_envs, env.device, write=write, **scope).items():
            say(f"  {path}: simulation_backend='auto' uses {pick} at {env.num_envs} environments")
    if (not settings or backends.RENDERER_SETTING in settings) and env.render_backends and env.has_dynamics:
        # A first draw registers the render paths of the class; then both renderers of this class are timed.
        env.render()
        mine = [
            path
            for path, _, _ in backends.registered(
                env.device, tuning.paths_of(type(env)), setting=backends.RENDERER_SETTING
            )
            if path.endswith(f"/{type(env).__name__}")
        ]
        render_paths = (
            backends.calibrate(env.num_envs, env.device, paths=mine, write=write, setting=backends.RENDERER_SETTING)
            if mine
            else {}
        )
        for path, pick in render_paths.items():
            say(f"  {path}: {env.num_envs} frames draw with the {pick} renderer")
    chosen = [k for k in settings_for(env) if not settings or k.name in settings]
    if not chosen:
        say(f"{type(env).__name__} reads no other tunable setting on {env.device.type}.")
        return {}
    paths = tuning.paths_of(type(env))
    try:
        saved = env.get_state() if env.has_dynamics else None
    except NotImplementedError:
        saved = None
    kwargs = dict(num_envs=env.num_envs, size=env.size, device=str(env.device))
    best, rows = {}, []
    try:
        for setting in chosen:
            path = next(p for p in paths if p in setting.paths)
            size = 0 if setting.step else env.size
            where = dict(path=path if setting.per_path else None, size=size, batch=env.num_envs, device=env.device)
            own = getattr(env, "_tuning_default", lambda *_: None)(setting.name, size)
            start = tuning.lookup(setting.name, measured=False, **where, **({} if own is None else {"default": own}))[
                0
            ]
            times, reference = {}, None
            for value in candidates(setting, start, env):
                with tuning.override(**best, **{setting.name: value}):
                    if setting.step:
                        with registry.make(type(env).__name__, **kwargs) as fresh:
                            seconds, result = _time(fresh, setting, repeats)
                    else:
                        seconds, result = _time(env, setting, repeats)
                if reference is not None and not _same(reference, result):
                    raise RuntimeError(
                        f"{setting.name}={value} changed the output of {type(env).__name__}; please report it"
                    )
                reference = result if reference is None else reference
                times[value] = seconds
            best[setting.name] = min(times, key=lambda v: (times[v], v != start))
            rows.append((setting, where, start, best[setting.name], times[start] / times[best[setting.name]]))
    finally:
        if saved is not None:
            env.set_state(saved)
    say(f"  {'setting':<22} {'default':>12} {'tuned':>12} {'speedup':>8}")
    for setting, _, start, pick, gain in rows:
        say(f"  {setting.name:<22} {start:>12} {pick:>12} {gain:>7.2f}x")
    if write:
        _save(env, rows, say)
    return best


def _save(env, rows, say) -> None:
    from . import registry

    files: dict[str, dict[str, dict[str, int]]] = {}
    for setting, where, _, pick, _ in rows:
        device = "cpu" if setting.device == "cpu" else str(env.device)
        key = (
            tuning.DEVICE_ENTRY
            if where["path"] is None
            else tuning.entry_key(where["path"], where["size"], env.num_envs)
        )
        files.setdefault(device, {}).setdefault(key, {})[setting.name] = pick
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


def representatives(device: str, skipped: dict[str, tuple[str, str]] | None = None) -> dict[str, str]:
    """Return one registered environment per tunable render path on ``device``.

    Classes whose extra is not installed are left out. ``skipped`` receives ``{path: (class, extra)}`` for the
    paths that only such classes cover.
    """
    from . import registry

    kind = torch.device(device).type
    wanted = {p for k in tuning.SETTINGS.values() if k.unit is not None and k.device in ("any", kind) for p in k.paths}
    out: dict[str, str] = {}
    missing: dict[str, tuple[str, str]] = {}
    for name in registry.list_envs():
        cls = registry.REGISTRY[name]
        if kind == "cpu" and getattr(cls, "requires_cuda", False):
            continue
        extra = registry.missing_extra(name)
        for path in tuning.paths_of(cls):
            if path in wanted:
                if extra is None:
                    out.setdefault(path, name)
                else:
                    missing.setdefault(path, (name, extra))
    if skipped is not None:
        skipped.update({path: entry for path, entry in missing.items() if path not in out})
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
            for key, settings in sorted(data.get("entries", {}).items()):
                print(f"  {key:<36} " + ", ".join(f"{k}={v}" for k, v in sorted(settings.items())))
    if not found:
        print("no tuning cached on this machine; every setting uses its derived default")
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
        skipped: dict[str, tuple[str, str]] = {}
        names = list(dict.fromkeys(representatives(device, skipped).values()))
        for name, extra in dict.fromkeys(skipped.values()):
            print(f"skipping {name}: needs the {extra} extra (pip install 'stable-envs[{extra}]')")
        print(f"tuning {len(names)} environments on {device}, about {len(names)} min: {', '.join(names)}")
        if not args.yes and (not sys.stdin.isatty() or input("start? [y/N] ").strip().lower() not in ("y", "yes")):
            print("not started; pass --yes to start without a prompt")
            return 1
    elif args.env:
        names = [args.env]
        try:
            extra = registry.missing_extra(args.env)
        except KeyError as e:
            print(f"stable-envs tune: {e.args[0]}", file=sys.stderr)
            return 2
        if extra is not None:
            print(
                f"stable-envs tune: {args.env} needs the {extra} extra: pip install 'stable-envs[{extra}]'",
                file=sys.stderr,
            )
            return 2
    else:
        print("name an environment, or pass --all, --show or --clear")
        return 2
    failed = []
    for name in names:
        kwargs = dict(num_envs=args.batch, device=device) | ({"size": args.size} if args.size else {})
        try:
            with registry.make(name, **kwargs) as env:
                tune(env, settings=args.setting or (), repeats=args.repeats)
        except (ImportError, RuntimeError) as e:
            if not args.all:
                if not isinstance(e, ImportError):
                    raise
                print(f"stable-envs tune: {name} cannot run here: {type(e).__name__}: {e}", file=sys.stderr)
                return 1
            print(f"skipping {name}: {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print(f"not tuned: {', '.join(failed)}")
        return 1
    return 0
