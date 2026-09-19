"""Explicit resource lifetime: the close protocol, the bounded LRU cache, and the never-release rule.

``render/_resources.py`` is the one Step-9 design change in the render backends, so what it promises
is tested against a FAKE backend object rather than against MJWarp or nvdiffrast. That is not a
convenience for a laptop with no CUDA: the promises are about ordering and idempotency — released
once, released before the object is reusable, never released twice, never released by the collector —
and a fake is the only way to observe them, because a real warp array's free is not visible from
Python at all.

The four backend classes are checked separately, by their SHAPE (they honour the protocol and declare
what they own) rather than by construction, which needs a GPU. The GPU arm asserts the counters
across repeated create/close cycles; see ``scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch``.
"""

from __future__ import annotations

import gc
import inspect
import threading

import pytest

from stable_envs.render import _resources as RES

# --------------------------------------------------------------------------- a fake backend object


class FakeBackend(RES.Closeable):
    """What a real backend does, with the frees replaced by an observable list."""

    resource_kind = "test.FakeBackend"

    def __init__(self, log: list, *, open_it: bool = True):
        self.log = log
        self.buffer = object()
        if open_it:
            self._opened()

    def _release(self) -> None:
        self.log.append("released")
        self.buffer = None

    def use(self):
        self._check_open("use")
        return self.buffer


# --------------------------------------------------------------------------- the close protocol


@pytest.mark.unit
def test_close_releases_once_and_is_idempotent():
    log = []
    b = FakeBackend(log)
    assert not b.closed
    b.close()
    b.close()
    b.close()
    assert log == ["released"], "close ran _release more than once"
    assert b.closed


@pytest.mark.unit
def test_use_after_close_raises_instead_of_reading_freed_state():
    b = FakeBackend([])
    assert b.use() is not None
    b.close()
    with pytest.raises(RES.UseAfterCloseError) as e:
        b.use()
    # the message has to name the object and say what close() did, because the caller sees it far
    # from the close that caused it
    assert "FakeBackend.use" in str(e.value) and "closed" in str(e.value)


@pytest.mark.unit
def test_the_context_manager_closes_on_the_way_out_including_on_an_exception():
    log = []
    with FakeBackend(log) as b:
        assert b.use() is not None
    assert log == ["released"] and b.closed

    log2 = []
    with pytest.raises(ValueError):
        with FakeBackend(log2):
            raise ValueError("boom")
    assert log2 == ["released"], "__exit__ must close on the error path too"


@pytest.mark.unit
def test_correctness_never_depends_on_del():
    """Dropping the last reference must NOT release: plan 2 forbids a GC-dependent lifetime.

    The test is the inverse of the usual one. A ``__del__`` that freed device memory would make every
    other guarantee here conditional on when the collector runs, and would run at interpreter shutdown
    after the CUDA bindings are gone. So the requirement is that an unclosed object leaks, visibly and
    countably, rather than being tidied up invisibly.
    """
    log = []
    FakeBackend(log)  # no reference kept
    gc.collect()
    gc.collect()
    assert log == [], "a backend object released itself from __del__ / GC"
    assert not hasattr(RES.Closeable, "__del__")


@pytest.mark.unit
def test_a_half_built_object_is_not_open():
    """``_opened`` is what declares an object live, so a constructor that raises leaves it closed."""
    b = FakeBackend([], open_it=False)
    assert b.closed
    with pytest.raises(RES.UseAfterCloseError):
        b.use()
    b.close()  # and closing it is a no-op, not a double free


@pytest.mark.unit
def test_the_counters_track_create_and_close_cycles():
    before = RES.counts()
    live0 = RES.live("test.FakeBackend")
    objs = [FakeBackend([]) for _ in range(5)]
    assert RES.live("test.FakeBackend") == live0 + 5
    for o in objs:
        o.close()
        o.close()  # idempotent: must not count twice
    assert RES.live("test.FakeBackend") == live0
    d = RES.delta(before)
    assert d["created"]["test.FakeBackend"] == 5
    assert d["closed"]["test.FakeBackend"] == 5


@pytest.mark.unit
def test_concurrent_close_releases_once():
    log = []
    b = FakeBackend(log)
    threads = [threading.Thread(target=b.close) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert log == ["released"]


@pytest.mark.unit
def test_a_failed_release_is_observable_and_not_counted_as_closed():
    class Boom(RES.Closeable):
        resource_kind = "test.Boom"

        def __init__(self):
            self.calls = 0
            self._opened()

        def _release(self):
            self.calls += 1
            raise RuntimeError("the free failed")

    before = RES.counts()
    live0 = RES.live("test.Boom")
    b = Boom()
    with pytest.raises(RuntimeError, match="the free failed"):
        b.close()
    assert b.closed and b.calls == 1
    assert RES.live("test.Boom") == live0 + 1
    change = RES.delta(before)
    assert change["created"]["test.Boom"] == 1
    assert change["release_failed"]["test.Boom"] == 1
    assert "test.Boom" not in change["closed"]
    assert RES.release_failures()[-1] == {"kind": "test.Boom", "error": "RuntimeError", "message": "the free failed"}
    b.close()
    assert b.calls == 1


# --------------------------------------------------------------------------- the bounded cache


@pytest.mark.unit
def test_the_cache_is_bounded():
    c = RES.BoundedCache(3, name="t")
    for i in range(50):
        c.put(i, i)
    assert len(c) == 3


@pytest.mark.unit
def test_eviction_is_least_recently_used_and_one_entry_at_a_time():
    """The property the ``clear()``-on-full idiom does not have, and the reason this class exists."""
    c = RES.BoundedCache(3, name="t")
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.get("a") == 1  # a is now the most recently USED
    c.put("d", 4)  # so b, the coldest, is what goes
    assert sorted(c.keys()) == ["a", "c", "d"]
    assert c.get("b") is None
    assert c.evictions == 1, "one insertion must evict one entry, not empty the cache"


@pytest.mark.unit
def test_a_key_in_use_survives_an_unrelated_insertion():
    """A clear()-on-full cache fails this, and that is the correctness half of the change.

    A caller holding the value for the key it is rendering with must not lose it because something
    else inserted. ``mjwarp_lightspec._CASTS_CACHE`` is the sharp case: the warm-up render writes the
    entry and the CAPTURING render reads it, and a miss there is a device-to-host copy that CUDA
    refuses inside a graph capture.
    """
    c = RES.BoundedCache(2, name="t")
    c.put("hot", "value")
    for i in range(20):
        c.get("hot")
        c.put(i, i)
    assert c.get("hot") == "value"


@pytest.mark.unit
def test_eviction_can_release_what_it_drops_and_a_replacement_counts_as_a_drop():
    log = []
    c = RES.BoundedCache(2, name="t", on_evict=lambda v: v.close())
    a, b, d = FakeBackend(log), FakeBackend(log), FakeBackend(log)
    c.put("a", a)
    c.put("b", b)
    c.put("c", d)  # evicts a
    assert a.closed and not b.closed and not d.closed
    c.put("b", FakeBackend(log))  # replacing b is as much a drop as evicting it
    assert b.closed


@pytest.mark.unit
def test_pop_hands_ownership_back_without_releasing():
    log = []
    c = RES.BoundedCache(2, name="t", on_evict=lambda v: v.close())
    b = FakeBackend(log)
    c.put("a", b)
    assert c.pop("a") is b
    assert not b.closed, "pop transfers ownership; only eviction and clear release"
    assert len(c) == 0


@pytest.mark.unit
def test_clear_releases_everything():
    log = []
    c = RES.BoundedCache(4, name="t", on_evict=lambda v: v.close())
    objs = [FakeBackend(log) for _ in range(3)]
    for i, o in enumerate(objs):
        c.put(i, o)
    c.clear()
    assert all(o.closed for o in objs) and len(c) == 0


@pytest.mark.unit
def test_a_failing_release_does_not_fail_the_insertion_that_triggered_it():
    def boom(_):
        raise RuntimeError("release failed")

    before = RES.counts()
    failures0 = len(RES.release_failures())
    c = RES.BoundedCache(1, name="t", on_evict=boom)
    c.put("a", 1)
    c.put("b", 2)  # must not raise: an eviction happens far from
    assert c.get("b") == 2  # anything the caller is thinking about
    assert RES.delta(before)["release_failed"]["cache:t"] == 1
    assert RES.release_failures()[failures0] == {
        "kind": "cache:t",
        "error": "RuntimeError",
        "message": "release failed",
    }


@pytest.mark.unit
def test_a_zero_capacity_cache_is_refused():
    with pytest.raises(ValueError):
        RES.BoundedCache(0)


# --------------------------------------------------------------------------- the deliberate leak


@pytest.mark.unit
def test_process_lifetime_retention_records_its_reason_and_is_capped():
    obj = object()
    assert RES.retain_process_lifetime(obj, "a unit test") is obj
    assert "a unit test" in RES.retained_reasons()


@pytest.mark.unit
def test_retention_past_the_cap_raises_rather_than_growing_without_limit(monkeypatch):
    monkeypatch.setattr(RES, "MAX_RETAINED", len(RES.retained_reasons()))
    with pytest.raises(ResourceWarning):
        RES.retain_process_lifetime(object(), "one too many")


@pytest.mark.unit
def test_a_foreign_keepalive_fallback_warns_and_retains_past_the_cap(monkeypatch, caplog):
    before = RES.counts()["retained"]
    monkeypatch.setattr(RES, "MAX_RETAINED", before)
    obj = object()
    assert RES.retain_process_lifetime(obj, "foreign texture owner", overflow="retain") is obj
    assert RES.counts()["retained"] == before + 1
    assert "Keeping it" in caplog.text


# --------------------------------------------------------------------------- the backends' shape


@pytest.mark.unit
@pytest.mark.parametrize(
    "module,cls",
    [
        ("stable_envs.render.mjwarp", "BatchRenderer"),
        ("stable_envs.render.nvdiff", "BatchRenderer"),
        ("stable_envs.render.nvdiff", "Scene"),
        ("stable_envs.render.rendernet", "LoadedCorrector"),
        ("stable_envs.render.rendernet", "_Runner"),
    ],
)
def test_every_backend_object_honours_the_close_protocol(module, cls):
    """Every backend declares the protocol; actual CPU construction is checked where available."""
    import importlib

    c = getattr(importlib.import_module(module), cls)
    assert issubclass(c, RES.Closeable), f"{module}.{cls} must be Closeable"
    assert c.resource_kind != RES.Closeable.resource_kind, f"{module}.{cls} needs its own kind"
    assert "_release" in c.__dict__, f"{module}.{cls} must say what it releases"
    assert "__del__" not in c.__dict__, f"{module}.{cls} must not free from the collector"


@pytest.mark.unit
def test_every_backend_construction_path_declares_itself_open():
    from stable_envs.render import mjwarp as MW
    from stable_envs.render import nvdiff as ND
    from stable_envs.render import rendernet as RN

    paths = (
        MW.BatchRenderer.__init__,
        ND.Scene.__init__,
        ND.Scene.from_triangles.__func__,
        ND.BatchRenderer.__init__,
        RN._Runner.__init__,
        RN.LoadedCorrector.__init__,
    )
    for path in paths:
        assert "._opened()" in inspect.getsource(path), f"{path.__qualname__} forgot _opened()"


@pytest.mark.unit
def test_a_cpu_scene_releases_every_owned_tensor_and_refuses_reuse():
    import numpy as np
    import torch

    from stable_envs.render import nvdiff as ND

    vert, norm, face = ND._box((0.5, 0.25, 1.0))
    before = RES.counts()
    live0 = RES.live("nvdiff.Scene")
    scene = ND.Scene.from_triangles(
        vert, norm, face, np.zeros(len(vert), np.int64), rgba=[[1.0, 1.0, 1.0, 1.0]], device="cpu"
    )
    owned = (
        "vert",
        "norm",
        "uv",
        "vgeom",
        "face",
        "fgeom",
        "mjgeom",
        "S3",
        "S4",
        "S3p",
        "mat_rgb",
        "alpha",
        "spec",
        "shininess",
        "emission",
        "texid",
    )
    assert all(isinstance(getattr(scene, name), torch.Tensor) for name in owned)
    assert RES.live("nvdiff.Scene") == live0 + 1
    scene.close()
    assert all(getattr(scene, name) is None for name in owned)
    assert scene.textures == {}
    assert RES.live("nvdiff.Scene") == live0
    change = RES.delta(before)
    assert change["created"]["nvdiff.Scene"] == 1
    assert change["closed"]["nvdiff.Scene"] == 1
    assert "nvdiff.Scene" not in change["release_failed"]
    with pytest.raises(RES.UseAfterCloseError, match=r"Scene\.clip"):
        scene.clip(None, None, None, None)


@pytest.mark.unit
def test_the_module_level_cast_memo_is_bounded():
    """``mjwarp_lightspec._CASTS_CACHE`` was an unbounded dict, i.e. one entry per model, forever."""
    pytest.importorskip("mujoco_warp")
    import stable_envs.render.mjwarp_lightspec as LS

    assert isinstance(LS._CASTS_CACHE, RES.BoundedCache)
    assert LS._CASTS_CACHE.capacity == LS.CASTS_CACHE_MAX


@pytest.mark.unit
def test_the_per_renderer_ray_cache_is_bounded():
    """``nvdiff.BatchRenderer._ray_cache``, checked without building a renderer (it needs CUDA)."""
    import stable_envs.render.nvdiff as ND

    assert ND.RAY_CACHE_MAX == 8, "the bound is the one the uncapped dict already used"
    src = _code_of("nvdiff.py")
    assert "_ray_cache = _resources . BoundedCache" in src
    assert "self . _ray_cache . put (" in src and "self . _ray_cache . get (" in src


def _code_of(name: str) -> str:
    """A module's source with every comment and string literal removed.

    A plain substring search over the file finds the PROSE — these modules explain at length why an
    EGL context is never freed — so the check has to look at code. ``tokenize`` is enough and needs no
    import of the module, which matters because three of the five need CUDA to import.
    """
    import pathlib
    import tokenize

    path = pathlib.Path(RES.__file__).parent / name
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mjwarp.py", "mjwarp_lightspec.py", "nvdiff.py", "rendernet.py", "_resources.py"])
def test_nothing_in_the_render_backends_releases_an_egl_context(name):
    """The rule ``testing/resource_guard.py`` enforces at zero tolerance, checked at the source.

    Releasing one EGL context corrupts rendering in every other live context in the process; four
    separate incidents in this repository were exactly that, each surfacing as pixel differences in an
    env that did nothing wrong. So ``close()`` on a render backend must never reach ``GLContext.free``
    or ``mujoco.Renderer.close``, however tempting "release what you acquired" reads — and this file
    is where a later reader who deletes that asymmetry as an oversight finds out.
    """
    code = _code_of(name)
    for forbidden in (". free (", "GLContext", "eglDestroyContext", "mujoco . Renderer"):
        assert forbidden not in code, f"{name} must not reach {forbidden!r}"


@pytest.mark.unit
def test_family_device_caches_release_on_eviction():
    """Workstream I's consumer list: overflow must call close/release, not drop to the collector."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "stable_envs"
    checks = [
        (root / "envs" / "fetch.py", 'BoundedCache(6, name="fetch.ctx"', "release_cached_ctx"),
        (root / "envs" / "ogbench" / "manipulation.py", 'BoundedCache(4, name="ogbench.ctx"', "release_cached_ctx"),
        (root / "envs" / "robocasa.py", 'BoundedCache(4, name="robocasa.ctx"', "release_cached_ctx"),
        (root / "envs" / "robocasa.py", "BoundedCache(", "robocasa.correctors"),
        (root / "envs" / "humanoidbench.py", 'BoundedCache(32, name="humanoidbench.ctx"', "release_cached_ctx"),
        (root / "envs" / "metaworld.py", 'BoundedCache(3, name="metaworld.ctx"', "release_cached_ctx"),
        (root / "envs" / "dmcontrol" / "_model.py", 'BoundedCache(3, name="dmcontrol.tc"', "release_cached_ctx"),
        (
            root / "envs" / "mujoco_locomotion.py",
            'BoundedCache(16, name="mujoco_locomotion.nvd"',
            "close_cached_renderer",
        ),
        (root / "envs" / "rocket_landing.py", 'BoundedCache(8, name="rocket.nvd"', "close_cached_renderer"),
        (root / "envs" / "rocket_landing.py", "owns_scene = True", "owns_scene"),
    ]
    for path, needle, label in checks:
        src = path.read_text(encoding="utf-8")
        assert needle in src, f"{path.relative_to(root.parent)} missing {label}: {needle!r}"
    robocasa = (root / "envs" / "robocasa.py").read_text(encoding="utf-8")
    assert "on_evict=lambda c: c.close()" in robocasa
    cpu = [
        (root / "envs" / "fetch.py", 'BoundedCache(32, name="fetch.plate"'),
        (root / "envs" / "fetch.py", 'BoundedCache(256, name="fetch.cam"'),
        (root / "envs" / "ogbench" / "maze.py", 'BoundedCache(_SCENE_CACHE_MAX, name="ogbench.maze"'),
        (root / "envs" / "ogbench" / "manipulation.py", 'BoundedCache(_PLATE_CACHE_MAX, name="ogbench.plate"'),
        (root / "envs" / "metaworld.py", 'BoundedCache(_PLATE_CACHE_MAX, name="metaworld.plate"'),
        (root / "envs" / "dmcontrol" / "_model.py", 'BoundedCache(_SCENE_CACHE_MAX, name="dmcontrol.scene"'),
        (root / "envs" / "mujoco_locomotion.py", 'BoundedCache(_SCENE_CACHE_MAX, name="mujoco_locomotion.scene"'),
        (root / "envs" / "maze2d.py", 'BoundedCache(_SCENE_CACHE_MAX, name="maze2d.scene"'),
        (root / "envs" / "classic_control.py", 'BoundedCache(_TERRAIN_CACHE_MAX, name="classic.terrain"'),
        (root / "envs" / "navigation.py", 'BoundedCache(_BAKE_CACHE_MAX, name="navigation.bake"'),
        (root / "envs" / "pgx.py", 'BoundedCache(_PLATE_CACHE_MAX, name="pgx.plate"'),
        (root / "render" / "raster2d.py", 'BoundedCache(_BASE_CACHE_MAX, name="raster2d.base"'),
        (root / "render" / "raycast3d.py", "_Memo = _RES.BoundedCache"),
        (root / "envs" / "craftax.py", 'BoundedCache(_MAX_ENTRIES, name="craftax.composite"'),
        (root / "envs" / "_craftax_full.py", 'BoundedCache(_MAX_ENTRIES, name="craftax_full.tables"'),
    ]
    for path, needle in cpu:
        src = path.read_text(encoding="utf-8")
        assert needle in src, f"{path.relative_to(root.parent)} missing CPU BoundedCache: {needle!r}"
