"""The contrib cases of ``tests/unit/test_render_resources.py``: the same checks, on the contrib render
engine (``contrib/render/lightspec.py``) and the contrib families' device and CPU caches."""

from __future__ import annotations

import tokenize
from pathlib import Path

import pytest

from stable_envs.render import _resources as RES

ROOT = Path(__file__).resolve().parents[3]
CONTRIB = ROOT / "contrib"


@pytest.mark.unit
def test_the_module_level_cast_memo_is_bounded():
    """``mjwarp_lightspec._CASTS_CACHE`` was an unbounded dict, i.e. one entry per model, forever."""
    pytest.importorskip("mujoco_warp")
    import contrib.render.lightspec as LS

    assert isinstance(LS._CASTS_CACHE, RES.BoundedCache)
    assert LS._CASTS_CACHE.capacity == LS.CASTS_CACHE_MAX


def _code_of(path: Path) -> str:
    """A module's source with every comment and string literal removed (see ``tests/unit/test_render_resources.py``)."""
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


@pytest.mark.unit
def test_nothing_in_the_render_backends_releases_an_egl_context():
    """The never-release rule of ``tests/unit/test_render_resources.py``, on the contrib render engine."""
    code = _code_of(CONTRIB / "render" / "lightspec.py")
    for forbidden in (". free (", "GLContext", "eglDestroyContext", "mujoco . Renderer"):
        assert forbidden not in code, f"lightspec.py must not reach {forbidden!r}"


@pytest.mark.unit
def test_family_device_caches_release_on_eviction():
    """The contrib rows of the renderer-resource consumer list: overflow must call close/release."""
    checks = [
        (CONTRIB / "robocasa.py", 'BoundedCache(4, name="robocasa.ctx"', "release_cached_ctx"),
        (CONTRIB / "humanoidbench.py", 'BoundedCache(32, name="humanoidbench.ctx"', "release_cached_ctx"),
        (CONTRIB / "metaworld.py", 'BoundedCache(3, name="metaworld.ctx"', "release_cached_ctx"),
        (
            CONTRIB / "mujoco_locomotion.py",
            'BoundedCache(16, name="mujoco_locomotion.nvd"',
            "close_cached_renderer",
        ),
    ]
    for path, needle, label in checks:
        src = path.read_text(encoding="utf-8")
        assert needle in src, f"{path.relative_to(ROOT)} missing {label}: {needle!r}"
    cpu = [
        (CONTRIB / "metaworld.py", 'BoundedCache(_PLATE_CACHE_MAX, name="metaworld.plate"'),
        (CONTRIB / "mujoco_locomotion.py", 'BoundedCache(_SCENE_CACHE_MAX, name="mujoco_locomotion.scene"'),
        (CONTRIB / "maze2d.py", 'BoundedCache(_SCENE_CACHE_MAX, name="maze2d.scene"'),
        (CONTRIB / "pgx.py", 'BoundedCache(_PLATE_CACHE_MAX, name="pgx.plate"'),
    ]
    for path, needle in cpu:
        src = path.read_text(encoding="utf-8")
        assert needle in src, f"{path.relative_to(ROOT)} missing CPU BoundedCache: {needle!r}"
