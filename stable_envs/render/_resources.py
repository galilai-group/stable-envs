"""Who owns a GPU resource, when it is released, and the one class of resource that never is.

Every backend object has explicit lifetime: ``close()``, a context manager, and no
correctness that depends on ``__del__``. This module is where that
rule is written down once, because the two backends it serves — ``mjwarp.py`` and ``nvdiff.py`` —
hold four different kinds of thing with three different correct answers, and getting the answer
wrong in either direction has already cost this repository days.

THE THREE ANSWERS
-----------------

1. **Exclusively-owned device memory is released on ``close()``.** An MJWarp ``Model``/``Data`` pair
   sized for one batch width, a render context's ``rgb_data``, a captured warp graph, a triangle
   soup, a texture pyramid, a CUDA-graph replay buffer. Exactly one object holds each, that object
   knows when it is finished with it, and dropping the last reference is what frees it. ``Closeable``
   below is the protocol: ``close()`` is idempotent, it runs ``_release()`` once, and every later
   call through the object raises :class:`UseAfterCloseError` instead of reading a freed pointer.

2. **Shared cached scenery is released by its CACHE, never by a user of it.** A baked plate, a
   prepared model bundle, a reference env, an ``mjw`` render context keyed by ``(cell, res, nworld)``
   — these are keyed, process-scoped and handed to every env that asks for the same key. An env that
   "released what it acquired" here would free a context another live env is still rendering through.
   So a consumer's ``_close`` drops its REFERENCE and nothing else; the cache is the owner, and
   :class:`BoundedCache` is how a cache bounds itself (LRU, one eviction at a time, an optional
   release callback run on the evicted value).

3. **An EGL/GL context is never released at all, deliberately.** This is not an oversight the library
   tidies up; it is a measured rule with a failure mode. ``stable_envs/testing/resource_guard.py``
   documents four separate incidents in which releasing ONE context silently corrupted rendering in
   every other live context in the process, surfacing minutes later as pixel differences in an env
   that did nothing wrong — and it fails any test that releases one, at zero tolerance. 96
   simultaneous contexts were measured to be harmless on this hardware. So the honest lifetime for an
   EGL context is "the process", ``close()`` must never free one, and :func:`retain_process_lifetime`
   is where anything in that class is parked with its reason recorded.

   The consequence for the public API is worth stating plainly, because a reader who sees
   ``env.close()`` will assume otherwise: **closing an env does not free its EGL context, and closing
   every env in a process does not return the process to zero contexts.** What ``close()`` buys is
   that the env's own device buffers go away at a point the caller chose, and that a closed env
   refuses to render rather than drawing through half-freed state.

WHAT THIS MODULE DOES NOT DO
----------------------------

It does not import torch, warp, mujoco or nvdiffrast, and it must not start: it is imported by
``render/`` modules that are themselves imported in interpreters where none of those exist, and the
dependency-direction test walks every file in ``render/`` to check it knows nothing about ``envs/``.
Everything here is stdlib and works on a fake backend object, which is also
how it is unit-tested on a machine with no CUDA.

It also does not count EGL contexts — ``testing/resource_guard.py`` does that, by patching mujoco's
constructors, and it is the authority on rule 3. :func:`counts` here is the complement: it counts
the backend objects THIS repository constructs, so a test can assert that ten create/close cycles
produced ten closes and left nothing live, which is the thing ``resource_guard`` cannot see.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any


class UseAfterCloseError(RuntimeError):
    """Raised when a released backend object is used again.

    A distinct type rather than a bare ``RuntimeError`` because it is the one failure this whole
    module exists to turn into a message instead of a wrong frame or a segfault, and a test that
    asserts the rule should be able to name it.
    """


# --------------------------------------------------------------------------- accounting

_LOCK = threading.Lock()
_CREATED: dict[str, int] = {}
_CLOSED: dict[str, int] = {}
_EVICTED: dict[str, int] = {}
_RELEASE_FAILED: dict[str, int] = {}
_RELEASE_FAILURES: list[dict[str, str]] = []
_RETAINED: list[tuple[str, Any]] = []
_LOG = logging.getLogger(__name__)

#: Cap on :func:`retain_process_lifetime`. Anything in that list is a deliberate leak, so the cap is
#: not a safety margin — it is a tripwire for a caller that retains per frame or per state instead of
#: per model. Nothing legitimate in this repository retains more than a handful.
MAX_RETAINED = 256


def _bump(d: dict[str, int], kind: str) -> None:
    with _LOCK:
        d[kind] = d.get(kind, 0) + 1


def counts() -> dict[str, Any]:
    """Resource counters since interpreter start, including failed releases.

    Cumulative and never reset, so a caller takes two snapshots and subtracts — the same shape
    ``testing/resource_guard.take``/``delta`` use, for the same reason: a test cares about what
    happened inside its own window, not about what the session did before it.
    """
    with _LOCK:
        return {
            "created": dict(_CREATED),
            "closed": dict(_CLOSED),
            "release_failed": dict(_RELEASE_FAILED),
            "evicted": dict(_EVICTED),
            "retained": len(_RETAINED),
        }


def live(kind: str) -> int:
    """How many objects of ``kind`` have been created and not yet closed."""
    with _LOCK:
        return _CREATED.get(kind, 0) - _CLOSED.get(kind, 0)


def delta(before: dict, after: dict | None = None) -> dict[str, dict[str, int]]:
    """What happened between two :func:`counts` snapshots, per kind."""
    after = counts() if after is None else after
    out: dict[str, dict[str, int]] = {}
    for section in ("created", "closed", "release_failed", "evicted"):
        a, b = after[section], before[section]
        keys = set(a) | set(b)
        out[section] = {k: a.get(k, 0) - b.get(k, 0) for k in sorted(keys) if a.get(k, 0) - b.get(k, 0)}
    out["retained"] = {"retained": after["retained"] - before["retained"]}
    return out


def release_failures() -> list[dict[str, str]]:
    """Details of failed releases, in occurrence order, for diagnostics."""
    with _LOCK:
        return [dict(x) for x in _RELEASE_FAILURES]


def _record_release_failure(kind: str, exc: BaseException) -> None:
    detail = {"kind": str(kind), "error": type(exc).__name__, "message": str(exc)}
    with _LOCK:
        _RELEASE_FAILED[kind] = _RELEASE_FAILED.get(kind, 0) + 1
        _RELEASE_FAILURES.append(detail)
    _LOG.warning("resource release failed for %s: %s: %s", kind, type(exc).__name__, exc)


def retain_process_lifetime(obj: Any, why: str, *, overflow: str = "raise") -> Any:
    """Keep ``obj`` alive until the interpreter exits, on purpose, with the reason recorded.

    Two things land here and nothing else should. The first is an EGL/GL context, for the reason in
    this module's docstring. The second is a device array handed to a foreign object that does NOT
    take ownership of it — ``warp.from_torch`` is the live example: a torch tensor given to a live
    ``wp.Texture2D`` and then dropped is freed under the texture and sampled as whatever the
    allocator put there next, which is a wrong picture with no error.

    Returns ``obj`` so it can be used inline. By default, raises past :data:`MAX_RETAINED`, because
    growth with frame count is a leak rather than a policy. ``overflow="retain"`` preserves an older
    unbounded keepalive fallback whose foreign object cannot accept an ownership attribute: it logs
    the overflow and retains anyway, because refusing there would free storage still in use.
    """
    if overflow not in ("raise", "retain"):
        raise ValueError(f"overflow must be 'raise' or 'retain', got {overflow!r}")
    with _LOCK:
        if len(_RETAINED) >= MAX_RETAINED:
            message = (
                f"retain_process_lifetime: {len(_RETAINED)} objects are already retained for the "
                f"life of the process (cap {MAX_RETAINED}); refusing to retain another for {why!r}. "
                f"Retention is per model or per context, never per frame or per state — a growing "
                f"count means the caller is retaining in a loop. Retained so far: "
                f"{sorted({w for w, _ in _RETAINED})}"
            )
            if overflow == "raise":
                raise ResourceWarning(message)
            _LOG.warning("%s Keeping it because this caller cannot transfer ownership.", message)
        _RETAINED.append((why, obj))
    return obj


def retained_reasons() -> list[str]:
    """The ``why`` of everything currently retained, for a report or a failure message."""
    with _LOCK:
        return [w for w, _ in _RETAINED]


# --------------------------------------------------------------------------- the protocol


class Closeable:
    """Explicit lifetime for a backend object: ``close()``, ``with``, and no reliance on ``__del__``.

    A subclass sets :attr:`resource_kind` (what the accounting calls it), calls
    :meth:`_opened` once it has actually acquired something, implements :meth:`_release` to drop
    exactly what it exclusively owns, and calls :meth:`_check_open` at the top of every method that
    touches a released resource.

    ``close()`` is idempotent and ``_release`` runs at most once: the second call is a no-op rather
    than a second free, which matters because the natural spelling at a call site is
    ``with env: ...`` inside a function that also closes on an error path.

    **There is deliberately no ``__del__``.** Adding one would make every rule in this module
    conditional on when the garbage collector happens to run, which is precisely the dependency
    plan 2 forbids; and a ``__del__`` on an object holding CUDA state runs at interpreter shutdown
    after the driver bindings have been torn down, which is the source of the "Exception ignored in"
    noise ``resource_guard`` documents. An object that is never closed leaks until the process ends,
    loudly countable via :func:`live`, which is a better failure than a nondeterministic one.
    """

    #: What :func:`counts` calls objects of this class. Subclasses override.
    resource_kind: str = "resource"

    _se_closed: bool = True  # until _opened() says otherwise: a half-built object is not open
    _se_counted: bool = False

    def _opened(self) -> None:
        """Declare the object live and count it. Called once the resources actually exist."""
        if not hasattr(self, "_se_close_lock"):
            self._se_close_lock = threading.Lock()
        self._se_closed = False
        if not self._se_counted:
            self._se_counted = True
            _bump(_CREATED, self.resource_kind)

    @property
    def closed(self) -> bool:
        return bool(self._se_closed)

    def _check_open(self, what: str) -> None:
        if self._se_closed:
            raise UseAfterCloseError(
                f"{type(self).__name__}.{what}: this renderer is closed. close() released its device "
                f"buffers, so what came back would be drawn through memory that no longer belongs to "
                f"it — or, worse, would look like a frame. Build a new one."
            )

    def _release(self) -> None:
        """Drop what this object EXCLUSIVELY owns. Never an EGL context, never a cached shared value."""

    def close(self) -> None:
        """Release this object's resources. Idempotent."""
        lock = getattr(self, "_se_close_lock", None)
        if lock is None:
            lock = self._se_close_lock = threading.Lock()
        with lock:
            if self._se_closed:
                return
            self._se_closed = True
            try:
                self._release()
            except Exception as exc:
                _record_release_failure(self.resource_kind, exc)
                raise
            if self._se_counted:
                _bump(_CLOSED, self.resource_kind)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# --------------------------------------------------------------------------- the cache


class BoundedCache:
    """A keyed cache with a hard capacity and LRU eviction, and an optional release of what it drops.

    It replaces the idiom this repository grew independently in a dozen modules::

        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()                  # <- throws away the entry in use, too
        _CACHE[key] = build(...)

    ``clear()``-on-full is not a bound with a different constant: it is a bound with a *cliff*. The
    entry a caller is holding right now is as likely to be discarded as the coldest one, so a
    working set at the capacity rebuilds everything on every miss (measured on
    OGBench's scene cache, which is why it uses an ``OrderedDict`` LRU, the policy generalised
    here), and — the reason this is a correctness class and not a performance one — a value that was
    handed out and is still being read can be dropped by an *unrelated* insertion.

    Eviction never frees anything by default. ``on_evict`` is how a cache whose values own device
    memory releases one, and the intended argument is ``lambda v: v.close()`` on a :class:`Closeable`.
    It is called outside the lock, after the entry is already out of the mapping, so a callback that
    itself touches the cache cannot deadlock or resurrect a half-evicted entry.

    Not a ``dict`` subclass and not a ``MutableMapping``: the surface is deliberately four methods, so
    that every use reads as a cache rather than as a mapping someone might mutate in place.
    """

    def __init__(self, capacity: int, *, name: str = "cache", on_evict: Callable[[Any], None] | None = None):
        if int(capacity) < 1:
            raise ValueError(f"BoundedCache({name!r}) needs capacity >= 1, got {capacity!r}")
        self.capacity = int(capacity)
        self.name = str(name)
        self.on_evict = on_evict
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key, default=None):
        """The value for ``key``, marking it most-recently-used so it is never the next evicted."""
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return default

    def put(self, key, value):
        """Insert ``key``, evicting the least recently used entries until the capacity holds.

        Returns ``value``, so a builder reads ``return cache.put(key, build())``. Re-putting an
        existing key replaces it and releases the OLD value if ``on_evict`` is set — a replacement is
        as much a drop as an eviction, and a cache that leaked on overwrite would leak silently.
        """
        drop = []
        with self._lock:
            if key in self._d:
                drop.append(self._d.pop(key))
            self._d[key] = value
            while len(self._d) > self.capacity:
                _, old = self._d.popitem(last=False)  # least recently used, ONE at a time
                self.evictions += 1
                _bump(_EVICTED, self.name)
                drop.append(old)
        for old in drop:
            self._free(old)
        return value

    def pop(self, key, default=None):
        """Remove ``key`` and RETURN its value without releasing it: the caller takes ownership."""
        with self._lock:
            return self._d.pop(key, default)

    def clear(self) -> None:
        """Drop every entry, releasing each through ``on_evict``. For teardown and for tests."""
        with self._lock:
            vals = list(self._d.values())
            self._d.clear()
        for v in vals:
            self._free(v)

    def _free(self, value) -> None:
        if self.on_evict is None:
            return
        try:
            self.on_evict(value)
        except Exception as exc:  # noqa: BLE001
            # A cache must not become the thing that fails. An eviction happens at an arbitrary
            # insertion, far from anything the caller is thinking about, so raising here would
            # attribute a release failure to whichever unrelated call happened to overflow it.
            # It still must be observable: otherwise a repo-wide cache can leak every value while
            # reporting successful eviction.
            _record_release_failure(f"cache:{self.name}", exc)

    def keys(self) -> list:
        with self._lock:
            return list(self._d.keys())

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._d

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def __repr__(self) -> str:
        return (
            f"BoundedCache({self.name!r}, {len(self)}/{self.capacity}, "
            f"hits={self.hits} misses={self.misses} evictions={self.evictions})"
        )
