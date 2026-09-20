"""A scene the caller is actively rendering must never be evicted from under it.

``OGBenchMaze._scene`` memoises baked plates. It used to drop the WHOLE cache on reaching its cap,
which meant that crossing the threshold discarded the key the current render was using and the next
render of the same appearance re-baked it. That would be free if a bake were a pure function of
(scene, style). It is not: measured on a 3090, three bakes of one key with nothing in between return
two distinct plates, 1/255 apart on four pixels
(``tests/fixtures/regression/join/step6/raycast3d/bake_probe_branch.json``). So a cache eviction could change the
frame a configuration was already producing — which is how ``check_style_is_state`` observed 1 on two
OGBench cells in Step 4 and 0 on nine that carry the identical defect.

These tests are about the CACHE POLICY, so they drive it with a stub instead of building an env: a
real one needs MuJoCo, an offscreen GL context and ~0.3 s per bake, and 300 bakes to prove a
one-line eviction rule is the wrong trade. The policy is the thing that regressed and the policy is
what is asserted.
"""

from __future__ import annotations

import pytest

OGB = pytest.importorskip("stable_envs.envs.ogbench")


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
    """The cache is module-level, so a test that leaves entries in it changes the next one."""
    saved = {k: OGB._SCENE_CACHE.get(k) for k in OGB._SCENE_CACHE.keys()}
    OGB._SCENE_CACHE.clear()
    try:
        yield
    finally:
        OGB._SCENE_CACHE.clear()
        for k, v in saved.items():
            OGB._SCENE_CACHE.put(k, v)


def test_a_hot_key_survives_far_more_distinct_inserts_than_the_cap():
    """Touch one key throughout; push 300 others past a cap of 256; it must never be re-baked."""
    cap = OGB._SCENE_CACHE_MAX
    stub = _Recorder()
    hot = _scene(stub, 224, "hot")
    assert stub.bakes == [(224, "hot")]

    for i in range(300):
        _scene(stub, 224, f"cold-{i}")
        got = _scene(stub, 224, "hot")  # the caller is still using it
        assert got is hot, (
            f"the hot scene was evicted and re-baked after {i + 1} distinct inserts (cap {cap}) — a "
            f"cache eviction can change the plate, so a key in use must not be the one dropped"
        )

    assert len(stub.bakes) == 301, (
        f"expected 1 hot bake + 300 cold ones, got {len(stub.bakes)}; the hot key was re-baked "
        f"{len(stub.bakes) - 301} time(s)"
    )
    assert len(OGB._SCENE_CACHE) <= cap, f"the cache holds {len(OGB._SCENE_CACHE)} entries against a cap of {cap}"


def test_eviction_drops_one_entry_and_not_the_whole_cache():
    """Crossing the cap must cost the LEAST RECENTLY USED key, not every key."""
    cap = OGB._SCENE_CACHE_MAX
    stub = _Recorder()
    for i in range(cap):
        _scene(stub, 224, f"k{i}")
    assert len(OGB._SCENE_CACHE) == cap

    _scene(stub, 224, "k0")  # k0 is now the most recently used
    _scene(stub, 224, "overflow")  # so k1 is the one that goes

    assert len(OGB._SCENE_CACHE) == cap, (
        f"after one insert past the cap the cache holds {len(OGB._SCENE_CACHE)} entries; a wholesale "
        f"clear() would have left 1"
    )
    assert stub._plate_key(224, "k0") in OGB._SCENE_CACHE, "the most recently used key was evicted"
    assert stub._plate_key(224, "k1") not in OGB._SCENE_CACHE, "the least recently used key survived"
    assert stub._plate_key(224, "overflow") in OGB._SCENE_CACHE


def test_the_common_case_never_evicts_anything():
    """One appearance rendered a thousand times is one bake. This is the case that pays for the cache."""
    stub = _Recorder()
    first = _scene(stub, 224, None)
    for _ in range(1000):
        assert _scene(stub, 224, None) is first
    assert stub.bakes == [(224, None)]
