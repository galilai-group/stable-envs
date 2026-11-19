"""Make the on-disk cache of Numba ``cache=True`` functions safe when several processes write it at once.

Numba saves a new signature of a cached function in two files: it reads the index, takes the first free data file
number, writes the index, then writes the data file. No lock spans these steps. When two processes save different
signatures of one function at the same time, both can take the same number: the index can then name the machine
code of one signature under the key of the other. The entry stays on disk, and every later process that loads it
adds an overload for the wrong signature. A call from another compiled function then fails with
``TypingError: Invalid use of type(CPUDispatcher(...)) with parameters (...)``.

``install`` makes each save hold an exclusive lock, and each load a shared lock, on one file in the cache directory.
It also makes a load whose machine code has another signature a cache miss, so Numba compiles the signature again
and writes over a bad entry.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

LOCK_NAME = "numba-cache.lock"


@contextlib.contextmanager
def _locked(directory: str | None, exclusive: bool) -> Iterator[None]:
    """Hold a lock on ``directory``; continue without one where the platform or the directory has no locks."""
    try:
        import fcntl

        fd = os.open(os.path.join(str(directory), LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o666)
    except (ImportError, OSError):
        yield
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)


def _requested_args(sig) -> tuple | None:
    if isinstance(sig, tuple):
        return sig
    args = getattr(sig, "args", None)
    return tuple(args) if args is not None else None


def install() -> None:
    """Patch Numba's cache classes once; does nothing without Numba."""
    try:
        from numba.core import caching
    except ImportError:
        return
    files = caching.IndexDataCacheFile
    if getattr(files, "_stable_envs_locked", False):
        return
    save, load, flush = files.save, files.load, files.flush

    def locked_save(self, key, data):
        with _locked(getattr(self, "_cache_path", None), exclusive=True):
            save(self, key, data)

    def locked_load(self, key):
        with _locked(getattr(self, "_cache_path", None), exclusive=False):
            return load(self, key)

    def locked_flush(self):
        with _locked(getattr(self, "_cache_path", None), exclusive=True):
            flush(self)

    files.save, files.load, files.flush = locked_save, locked_load, locked_flush
    files._stable_envs_locked = True

    load_overload = caching.Cache._load_overload

    def checked_load_overload(self, sig, target_context):
        data = load_overload(self, sig, target_context)
        found = getattr(getattr(data, "signature", None), "args", None)
        wanted = _requested_args(sig)
        if found is not None and wanted is not None and tuple(found) != wanted:
            return None
        return data

    caching.Cache._load_overload = checked_load_overload
