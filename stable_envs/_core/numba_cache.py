"""Keep the on-disk cache of Numba ``cache=True`` functions correct when several processes write it at once.

To save a new signature, Numba reads the index, takes the first free data file number, writes the index and then
writes the data file, with no lock between processes. Two processes that save two signatures of one function at the
same time can take the same number. The index then gives the machine code of one signature to the other, and every
later process that loads that entry fails with ``TypingError: Invalid use of type(CPUDispatcher(...))``.

``install`` makes a save hold an exclusive lock, and a load a shared lock, on a file in the cache directory. A load
that returns the machine code of another signature is a cache miss, so Numba compiles again and replaces the entry.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import Iterator

LOCK_NAME = "numba-cache.lock"


@contextlib.contextmanager
def _locked(directory: str, exclusive: bool) -> Iterator[None]:
    """Hold a lock on ``directory``; continue without one where the platform or the directory has no locks."""
    try:
        import fcntl

        fd = os.open(os.path.join(directory, LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o666)
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
    """Patch the cache classes of Numba, and of Numba CUDA when ``numba.cuda`` is imported; once each.

    Does nothing without Numba. Call it again after ``import numba.cuda`` to cover the CUDA cache.
    """
    try:
        from numba.core import caching
    except ImportError:
        return
    _patch(caching)
    if "numba.cuda" in sys.modules:
        try:
            cuda_caching = importlib.import_module("numba.cuda.core.caching")
        except ImportError:
            return
        _patch(cuda_caching)


def _patch(caching) -> None:
    files = caching.IndexDataCacheFile
    if getattr(files, "_stable_envs_locked", False):
        return
    save, load, flush = files.save, files.load, files.flush

    def locked_save(self, key, data):
        with _locked(self._cache_path, exclusive=True):
            save(self, key, data)

    def locked_load(self, key):
        with _locked(self._cache_path, exclusive=False):
            return load(self, key)

    def locked_flush(self):
        with _locked(self._cache_path, exclusive=True):
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
