"""A baked maze plate that the caller is rendering is never evicted from under it.

A bake is not a pure function of (scene, style): repeated bakes of one key can differ by 1/255 on a
few pixels, so evicting a key in use could change a frame already being produced. The tests drive
the cache policy with a stub instead of building an env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensordict")
from stable_envs.envs.ogbench import maze as OGB  # noqa: E402


class _Recorder:
    """Enough of an ``OGBenchMaze`` for ``_scene`` to run: a key and a bake that counts itself."""

    def __init__(self):
        self.bakes = []

    def _plate_key(self, size, variation):
        return ("stub", int(size), variation)

    def _bake(self, size, variation):
        self.bakes.append((int(size), variation))
        return f"plate<{size},{variation}>"


def _scene(stub, size, variation):
    return OGB.OGBenchMaze._scene(stub, size, variation)


@pytest.fixture(autouse=True)
def _isolated_cache():
    """The cache is module-level; restore it so tests do not leak entries."""
    saved = {k: OGB._SCENE_CACHE.get(k) for k in OGB._SCENE_CACHE.keys()}
    OGB._SCENE_CACHE.clear()
    try:
        yield
    finally:
        OGB._SCENE_CACHE.clear()
        for k, v in saved.items():
            OGB._SCENE_CACHE.put(k, v)


def test_a_hot_key_survives_far_more_distinct_inserts_than_the_cap():
    """Touch one key throughout while 300 others pass the cap; it is never re-baked."""
    cap = OGB._SCENE_CACHE_MAX
    stub = _Recorder()
    hot = _scene(stub, 224, "hot")
    assert stub.bakes == [(224, "hot")]

    for i in range(300):
        _scene(stub, 224, f"cold-{i}")
        got = _scene(stub, 224, "hot")  # the caller is still using it
        assert got is hot, f"the hot scene was evicted and re-baked after {i + 1} distinct inserts (cap {cap})"

    assert len(stub.bakes) == 301, "1 hot bake + 300 cold ones; the hot key was re-baked"
    assert len(OGB._SCENE_CACHE) <= cap, f"the cache holds {len(OGB._SCENE_CACHE)} entries against a cap of {cap}"


def test_eviction_drops_one_entry_and_not_the_whole_cache():
    """Crossing the cap evicts the least recently used key and nothing else."""
    cap = OGB._SCENE_CACHE_MAX
    stub = _Recorder()
    for i in range(cap):
        _scene(stub, 224, f"k{i}")
    assert len(OGB._SCENE_CACHE) == cap

    _scene(stub, 224, "k0")  # k0 is now the most recently used
    _scene(stub, 224, "overflow")  # so k1 is the one that goes

    assert len(OGB._SCENE_CACHE) == cap, (
        f"after one insert past the cap the cache holds {len(OGB._SCENE_CACHE)} entries, not {cap}"
    )
    assert stub._plate_key(224, "k0") in OGB._SCENE_CACHE, "the most recently used key was evicted"
    assert stub._plate_key(224, "k1") not in OGB._SCENE_CACHE, "the least recently used key survived"
    assert stub._plate_key(224, "overflow") in OGB._SCENE_CACHE
