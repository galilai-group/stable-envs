"""Numba ``cache=True`` files stay correct when several processes compile the same kernels at once."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
import threading

import pytest

pytest.importorskip("fcntl")
caching = pytest.importorskip("numba.core.caching")

from stable_envs._core import numba_cache  # noqa: E402

WAIT = 1.0


def _cache_file(path):
    return caching.IndexDataCacheFile(cache_path=str(path), filename_base="kernel-1.py3", source_stamp=(1.0, 1))


def test_two_saves_at_once_keep_each_signature_with_its_own_data(tmp_path):
    """Two saves interleave as in the race: both read the empty index, then the last index and data differ."""
    numba_cache.install()
    first, second = _cache_file(tmp_path), _cache_file(tmp_path)
    second_loaded, first_indexed, second_saved = threading.Event(), threading.Event(), threading.Event()

    def hook(cache_file, name, before=None, after=None):
        original = getattr(cache_file, name)

        def wrapped(*args):
            if before is not None:
                before()
            out = original(*args)
            if after is not None:
                after()
            return out

        setattr(cache_file, name, wrapped)

    # Without a lock across a save: first reads, second reads, first writes the index, second writes the index
    # and its data, first writes its data into the file number that the index now gives to the second key.
    hook(first, "_save_index", before=lambda: second_loaded.wait(WAIT), after=first_indexed.set)
    hook(first, "_save_data", before=lambda: second_saved.wait(WAIT))
    hook(second, "_load_index", after=second_loaded.set)
    hook(second, "_save_index", before=lambda: first_indexed.wait(WAIT))
    hook(second, "_save_data", after=second_saved.set)
    threads = [
        threading.Thread(target=first.save, args=(("first",), "first data")),
        threading.Thread(target=second.save, args=(("second",), "second data")),
    ]
    threads[0].start()
    threads[1].start()
    for thread in threads:
        thread.join(10 * WAIT)
        assert not thread.is_alive()

    reader = _cache_file(tmp_path)
    index = reader._load_index()
    assert index
    for key, data_name in index.items():
        assert reader._load_data(data_name) == f"{key[0]} data"


KERNEL = textwrap.dedent(
    """
    from numba import njit

    @njit(cache=True)
    def total(a):
        return a.sum()
    """
)

# One signature for a contiguous array and one for a strided view of it.
COMPILE = "import numpy as np; import kernel; a = np.arange(6.0); kernel.total(a); kernel.total(a[::2])"

# A call from another compiled function that types the strided signature first.
CALL = textwrap.dedent(
    """
    import stable_envs._core.numba_cache as numba_cache

    numba_cache.install()
    import numpy as np
    from numba import njit

    import kernel

    @njit
    def caller(a):
        return kernel.total(a)

    a = np.arange(6.0)
    assert caller(a[::2]) == 6.0
    assert caller(a) == 15.0
    """
)


def _run(code, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(cwd), *sys.path]))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBA_DISABLE_JIT", None)
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_a_cache_entry_with_another_signature_compiles_again(tmp_path):
    """A cache entry that the race left on disk does not fail the processes that load it."""
    (tmp_path / "kernel.py").write_text(KERNEL)
    done = _run(COMPILE, tmp_path)
    assert done.returncode == 0, done.stderr
    (index_path,) = (tmp_path / "__pycache__").glob("kernel.total-*.nbi")
    with open(index_path, "rb") as f:
        version = pickle.load(f)
        stamp, overloads = pickle.loads(f.read())
    assert len(overloads) == 2
    keys = list(overloads)
    overloads[keys[0]], overloads[keys[1]] = overloads[keys[1]], overloads[keys[0]]
    with open(index_path, "wb") as f:
        pickle.dump(version, f, protocol=-1)
        f.write(pickle.dumps((stamp, overloads), protocol=-1))

    done = _run(CALL, tmp_path)
    assert done.returncode == 0, done.stderr
