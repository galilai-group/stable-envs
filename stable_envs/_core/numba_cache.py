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


def _open_lock(path: str) -> int | None:
    for flags in (os.O_RDWR | os.O_CREAT, os.O_RDONLY):
        try:
            return os.open(path, flags, 0o666)
        except OSError:
            continue
    return None


@contextlib.contextmanager
def _locked(directory: str, exclusive: bool) -> Iterator[None]:
    """Hold a lock on ``directory``; continue without one where the platform or the file system has no locks."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    fd = None if fcntl is None else _open_lock(os.path.join(directory, LOCK_NAME))
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError:
            os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)


def _requested_args(sig) -> tuple | None:
    if isinstance(sig, tuple):
        return sig
    args = getattr(sig, "args", None)
    if args is not None:
        return tuple(args)
    try:
        from numba.core import sigutils

        return tuple(sigutils.normalize_signature(sig)[0])
    except Exception:  # noqa: BLE001 — an unknown form skips the check
        return None


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
    if not getattr(files, "_stable_envs_locked", False):
        _lock_files(files)
    if not getattr(caching.Cache, "_stable_envs_checked", False):
        _check_loads(caching.Cache)


def _lock_files(files) -> None:
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


def _check_loads(cache) -> None:
    load_overload = cache._load_overload

    def checked_load_overload(self, sig, target_context):
        data = load_overload(self, sig, target_context)
        found = getattr(getattr(data, "signature", None), "args", None)
        wanted = _requested_args(sig)
        if found is not None and wanted is not None and tuple(found) != wanted:
            return None
        return data

    cache._load_overload = checked_load_overload
    cache._stable_envs_checked = True
