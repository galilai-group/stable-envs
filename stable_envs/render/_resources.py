"""Explicit ownership and accounting for renderer resources.

Instances release exclusively owned buffers on close. Caches own shared values
and choose whether eviction releases them. EGL/GL contexts remain alive for the
process lifetime: destroying one can corrupt other live contexts on supported
drivers. Foreign views also require their backing storage to outlive every use.

These helpers use only the standard library and can run without a GPU backend.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

from .._core.config import log as _log


class UseAfterCloseError(RuntimeError):
    """A released backend object was used again."""


_LOCK = threading.Lock()
_CREATED: dict[str, int] = {}
_CLOSED: dict[str, int] = {}
_EVICTED: dict[str, int] = {}
_RELEASE_FAILED: dict[str, int] = {}
_RELEASE_FAILURES: list[dict[str, str]] = []
_RETAINED: list[tuple[str, Any]] = []

#: Detect accidental per-frame retention of resources intended to be held per model or context.
MAX_RETAINED = 256


def _bump(d: dict[str, int], kind: str) -> None:
    with _LOCK:
        d[kind] = d.get(kind, 0) + 1


def counts() -> dict[str, Any]:
    """Return cumulative resource counters; use two snapshots to measure a time window."""
    with _LOCK:
        return {
            "created": dict(_CREATED),
            "closed": dict(_CLOSED),
            "release_failed": dict(_RELEASE_FAILED),
            "evicted": dict(_EVICTED),
            "retained": len(_RETAINED),
        }


def live(kind: str) -> int:
    """Return the number of open objects; release failures are counted separately."""
    with _LOCK:
        return _CREATED.get(kind, 0) - _CLOSED.get(kind, 0)


def delta(before: dict, after: dict | None = None) -> dict[str, dict[str, int]]:
    """Subtract resource counters between snapshots, defaulting the second to now."""
    after = counts() if after is None else after
    out: dict[str, dict[str, int]] = {}
    for section in ("created", "closed", "release_failed", "evicted"):
        a, b = after[section], before[section]
        keys = set(a) | set(b)
        out[section] = {k: a.get(k, 0) - b.get(k, 0) for k in sorted(keys) if a.get(k, 0) - b.get(k, 0)}
    out["retained"] = {"retained": after["retained"] - before["retained"]}
    return out


def release_failures() -> list[dict[str, str]]:
    """Return copies of recorded release failures in occurrence order."""
    with _LOCK:
        return [dict(x) for x in _RELEASE_FAILURES]


def _record_release_failure(kind: str, exc: BaseException) -> None:
    detail = {"kind": str(kind), "error": type(exc).__name__, "message": str(exc)}
    with _LOCK:
        _RELEASE_FAILED[kind] = _RELEASE_FAILED.get(kind, 0) + 1
        _RELEASE_FAILURES.append(detail)
    _log("WARNING", f"Resource release failed for {kind}: {type(exc).__name__}: {exc}")


def retain_process_lifetime(obj: Any, why: str, *, overflow: str = "raise") -> Any:
    """Keep an object alive until interpreter exit and return it.

    Record ``why`` for diagnostics. At MAX_RETAINED, the default raises
    RuntimeError; ``overflow="retain"`` logs and retains anyway for foreign
    views whose backing storage cannot safely be freed.
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
                raise RuntimeError(message)
            _log("WARNING", f"{message} Keeping it because this caller cannot transfer ownership.")
        _RETAINED.append((why, obj))
    return obj


def retained_reasons() -> list[str]:
    """Return the recorded reason for each process-lifetime reference."""
    with _LOCK:
        return [w for w, _ in _RETAINED]


class Closeable:
    """Explicit, idempotent resource lifetime with context-manager support.

    Subclasses call ``_opened`` after acquisition, implement ``_release`` for owned
    resources, and check ``_check_open`` before use. Close attempts release once;
    a failed release is recorded and re-raised, and the object remains closed.
    There is no finalizer: callers must close explicitly.
    """

    #: Accounting name for instances of this class.
    resource_kind: str = "resource"

    _se_closed: bool = True  # Resources are unusable until acquisition completes.
    _se_counted: bool = False

    def _opened(self) -> None:
        """Mark acquired resources live and create the close lock."""
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
                f"{type(self).__name__}.{what}: this renderer is closed. "
                "Create a new renderer before using its released buffers."
            )

    def _release(self) -> None:
        """Release exclusively owned resources; leave shared caches and EGL contexts alive."""

    def close(self) -> None:
        """Attempt release once, recording and re-raising any release failure."""
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
            finally:
                if self._se_counted:
                    _bump(_CLOSED, self.resource_kind)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class BoundedCache:
    """Thread-safe bounded LRU mapping with an optional eviction callback.

    Callbacks run outside the lock on replacement, eviction, or clear. They may
    release a value still referenced by a consumer; callers must coordinate its
    lifetime. Callback failures are recorded rather than raised by cache operations.
    ``pop`` transfers ownership without invoking the callback.
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
        """Return a value and mark it most recently used, or return ``default`` on a miss."""
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return default

    def put(self, key, value):
        """Insert and return a value, evicting the least recently used entry if needed.

        Replacing an existing key also invokes the release callback on its old value.
        """
        drop = []
        with self._lock:
            if key in self._d:
                drop.append(self._d.pop(key))
            self._d[key] = value
            while len(self._d) > self.capacity:
                _, old = self._d.popitem(last=False)
                self.evictions += 1
                _bump(_EVICTED, self.name)
                drop.append(old)
        for old in drop:
            self._free(old)
        return value

    def pop(self, key, default=None):
        """Remove and return a value without releasing it; the caller takes ownership."""
        with self._lock:
            return self._d.pop(key, default)

    def clear(self) -> None:
        """Remove all entries and invoke the release callback on each value."""
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
            # Keep eviction failures observable without making an unrelated cache insertion raise.
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
