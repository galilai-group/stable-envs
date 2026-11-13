"""Resource accounting for tests that keep GL contexts alive for the process lifetime.

Install constructor/release wrappers before environments are built and re-arm
after new backends load. Snapshot deltas enforce per-test allocation limits;
session totals detect runaway allocation. These are suite policies, not a
general proof that a GL resource is safe to release.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from dataclasses import dataclass, field

# Retain the earliest allocation sites; bounded storage avoids growth during long runs.
MAX_SITES = 512

# Limit stack capture overhead while retaining callers that identify the allocation path.
SITE_DEPTH = 12


@dataclass
class Counts:
    """Cumulative creation/release counts with a bounded list of call sites."""

    context_created: int = 0
    context_freed: int = 0
    renderer_created: int = 0
    renderer_closed: int = 0
    sites: list = field(default_factory=list)

    @property
    def context_live(self) -> int:
        return self.context_created - self.context_freed


_LOCK = threading.Lock()
_COUNTS = Counts()
_PEAK_LIVE = 0
_INSTALLED = False
_PATCHED: list[str] = []


def _record(kind: str, action: str) -> None:
    global _PEAK_LIVE
    with _LOCK:
        setattr(_COUNTS, f"{kind}_{action}", getattr(_COUNTS, f"{kind}_{action}") + 1)
        if _COUNTS.context_live > _PEAK_LIVE:
            _PEAK_LIVE = _COUNTS.context_live
        if len(_COUNTS.sites) < MAX_SITES:
            # Omit accounting frames so diagnostics identify the caller.
            stack = traceback.extract_stack(limit=SITE_DEPTH + 2)[:-2]
            _COUNTS.sites.append((f"{kind}.{action}", [f"{f.filename}:{f.lineno} {f.name}" for f in stack[-6:]]))


def _count_one(obj, kind: str, action: str) -> None:
    if action in ("freed", "closed"):
        # Explicit free and destructor paths can reach the same object; count its release once.
        try:
            if getattr(obj, "_se_released", False):
                return
            obj._se_released = True
        except Exception:  # noqa: BLE001
            pass  # __slots__ or a C type: fall back to counting
    _record(kind, action)


def _wrap(cls, name: str, kind: str, action: str) -> None:
    """Count successful calls to a method; repeated installation does not wrap it twice."""
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_se_guarded", False):
        return

    def wrapper(self, *a, __fn=fn, __kind=kind, __action=action, **k):
        out = __fn(self, *a, **k)
        try:
            _count_one(self, __kind, __action)
        except Exception:  # noqa: BLE001
            # Accounting can run during interpreter shutdown; it must not mask the wrapped result.
            pass
        return out

    wrapper._se_guarded = True  # type: ignore[attr-defined]
    wrapper.__name__ = name
    wrapper.__qualname__ = f"{cls.__name__}.{name}"
    setattr(cls, name, wrapper)
    _PATCHED.append(f"{cls.__module__}.{cls.__name__}.{name}")


def _wrap_dm_control() -> None:
    """Patch the loaded dm_control base class to count every backend and release path.

    Do not import dm_control here: unrelated tests must not initialize its display.
    """
    mod = sys.modules.get("dm_control._render.base")
    cls = getattr(mod, "ContextBase", None) if mod is not None else None
    if cls is None:
        return
    _wrap(cls, "__init__", "context", "created")
    _wrap(cls, "_free_unconditionally", "context", "freed")


def install() -> list[str]:
    """Install available MuJoCo wrappers and wrappers for already-loaded dm_control contexts.

    Safe to repeat as families load. Try importing MuJoCo and its GL backends;
    return the installed wrapper names even when some dependencies are unavailable.
    """
    global _INSTALLED
    _INSTALLED = True
    _wrap_dm_control()
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return list(_PATCHED)

    wrap = _wrap

    # Patch each available backend, deduplicating aliases such as mujoco.GLContext.
    seen = set()
    for modname in ("egl", "osmesa", "glfw"):
        try:
            mod = __import__(f"mujoco.{modname}", fromlist=["GLContext"])
        except Exception:  # noqa: BLE001
            continue
        cls = getattr(mod, "GLContext", None)
        if cls is None or id(cls) in seen:
            continue
        seen.add(id(cls))
        wrap(cls, "__init__", "context", "created")
        wrap(cls, "free", "context", "freed")
    cls = getattr(mujoco, "GLContext", None)
    if cls is not None and id(cls) not in seen:
        wrap(cls, "__init__", "context", "created")
        wrap(cls, "free", "context", "freed")

    # Track renderer churn separately from the contexts its constructors already count.
    rcls = getattr(mujoco, "Renderer", None)
    if rcls is not None:
        wrap(rcls, "__init__", "renderer", "created")
        wrap(rcls, "close", "renderer", "closed")
    return list(_PATCHED)


def take() -> Counts:
    """Return an independent counter snapshot."""
    with _LOCK:
        return Counts(
            _COUNTS.context_created,
            _COUNTS.context_freed,
            _COUNTS.renderer_created,
            _COUNTS.renderer_closed,
            list(_COUNTS.sites),
        )


def delta(before: Counts, after: Counts | None = None) -> dict:
    """Return count differences and call sites added between two snapshots."""
    after = take() if after is None else after
    return {
        "context_created": after.context_created - before.context_created,
        "context_freed": after.context_freed - before.context_freed,
        "renderer_created": after.renderer_created - before.renderer_created,
        "renderer_closed": after.renderer_closed - before.renderer_closed,
        "context_live": after.context_live,
        "sites": after.sites[len(before.sites) :],
    }


def peak_live() -> int:
    return _PEAK_LIVE


def totals() -> dict:
    c = take()
    return {
        "context_created": c.context_created,
        "context_freed": c.context_freed,
        "context_live": c.context_live,
        "context_peak_live": _PEAK_LIVE,
        "renderer_created": c.renderer_created,
        "renderer_closed": c.renderer_closed,
        "patched": list(_PATCHED),
    }


def format_sites(sites, limit: int = 3) -> str:
    out = []
    for tag, frames in sites[:limit]:
        out.append(f"    {tag}")
        out.extend(f"      {f}" for f in frames[-4:])
    if len(sites) > limit:
        out.append(f"    ... and {len(sites) - limit} more")
    return "\n".join(out)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    """Return per-test and session context limits, overridable through SE_MAX_* variables.

    The session limit scales with registry size because reference and plate
    renderers retain contexts. Per-test limits catch allocation proportional
    to calls or batch size; any release is a separate policy violation.
    """
    try:  # import here: this module must not need the registry
        from stable_envs import REGISTRY

        n_envs = len(REGISTRY)
    except Exception:  # noqa: BLE001
        n_envs = 0
    # One per env, plus reference envs and plate renderers for the families that bake them, plus slack.
    scaled = max(192, 2 * n_envs + 64)
    return {
        "per_test": _int_env("SE_MAX_CONTEXTS_PER_TEST", 8),
        "session_live": _int_env("SE_MAX_LIVE_CONTEXTS", scaled),
    }


def violations(d: dict, batch: int | None = None) -> list[tuple[str, str]]:
    """Return named policy violations with diagnostics for a test delta and session totals."""
    lim = limits()
    out = []
    if d["context_freed"]:
        out.append(
            (
                "EGL_CONTEXT_RELEASED",
                (
                    f"this test released {d['context_freed']} EGL context(s). Releasing one silently corrupts "
                    f"rendering in every other live context in the process, showing up later as pixel "
                    f"differences in an unrelated env. Cache the object for the life of the process; do not "
                    f"close it.\n"
                    f"  released at:\n{format_sites([s for s in d['sites'] if s[0].endswith('freed')])}"
                ),
            )
        )
    if d["renderer_closed"]:
        out.append(
            (
                "MUJOCO_RENDERER_CLOSED",
                (
                    f"this test closed {d['renderer_closed']} mujoco.Renderer(s). A Renderer owns an EGL "
                    f"context, so closing one is the same defect as EGL_CONTEXT_RELEASED; a common cause is "
                    f"constructing and closing a renderer per baked plate.\n"
                    f"  closed at:\n{format_sites([s for s in d['sites'] if s[0].endswith('closed')])}"
                ),
            )
        )
    n = d["context_created"]
    if batch and batch >= 4 and n >= batch:
        out.append(
            (
                "EGL_CONTEXTS_SCALE_WITH_BATCH",
                (
                    f"this test created {n} EGL contexts with a batch of {batch}: a per-environment value is "
                    f"being turned into a per-environment scene, e.g. a B-wide ramp of style values re-baking "
                    f"a MuJoCo plate per row. Two distinct values suffice; see check_per_env_variation and "
                    f"uniform_style.\n"
                    f"  created at:\n{format_sites([s for s in d['sites'] if s[0].endswith('created')])}"
                ),
            )
        )
    elif n > lim["per_test"]:
        out.append(
            (
                "EGL_CONTEXTS_PER_TEST",
                (
                    f"this test created {n} EGL contexts, over the limit of {lim['per_test']} "
                    f"(SE_MAX_CONTEXTS_PER_TEST). The most a test should need is one cached ogbench env, its "
                    f"plate renderers and one reference env; check for creation per state, style value or "
                    f"call.\n"
                    f"  created at:\n{format_sites([s for s in d['sites'] if s[0].endswith('created')])}"
                ),
            )
        )
    if d["context_live"] > lim["session_live"]:
        out.append(
            (
                "EGL_CONTEXTS_RUNAWAY",
                (
                    f"{d['context_live']} EGL contexts are live, over {lim['session_live']} "
                    f"(SE_MAX_LIVE_CONTEXTS); something is creating contexts in a loop."
                ),
            )
        )
    return out
