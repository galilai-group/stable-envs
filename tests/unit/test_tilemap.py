import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from stable_envs.render import tilemap
from stable_envs.render._cache import cache_path


@pytest.mark.parametrize("triton_available", [False, True])
def test_cpu_tiles_and_visibility_ignore_cuda_backend(monkeypatch, triton_available):
    monkeypatch.setattr(tilemap, "HAVE_TRITON", triton_available)
    atlas = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    indices = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected = torch.cat([atlas[1], atlas[0]], dim=1).unsqueeze(0)
    out = torch.empty_like(expected)
    assert tilemap.blit(atlas, indices, 2, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    transparent = torch.zeros(2, 3, 3, dtype=torch.bool)
    transparent[1] = True
    visible = tilemap.visibility_scan(transparent, 3)
    expected_visible = torch.zeros_like(transparent)
    expected_visible[0, 1, 2] = True
    expected_visible[1] = True
    assert torch.equal(visible, expected_visible)


def test_cache_location_respects_explicit_and_xdg_roots(monkeypatch, tmp_path):
    monkeypatch.delenv("STABLE_ENVS_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_path("tilemap", "a.npz") == tmp_path / "xdg/stable-envs/tilemap/a.npz"
    monkeypatch.setenv("STABLE_ENVS_CACHE", str(tmp_path / "explicit"))
    assert cache_path("tilemap", "a.npz") == tmp_path / "explicit/tilemap/a.npz"


@pytest.fixture(scope="module")
def official_tiles():
    pytest.importorskip("minigrid")
    return tilemap._official_atlas(2), tilemap._label_maps(2)


@pytest.fixture
def atlas_cache(monkeypatch, tmp_path, official_tiles):
    monkeypatch.setenv("STABLE_ENVS_CACHE", str(tmp_path))
    monkeypatch.setattr(tilemap, "_official_atlas", lambda size: official_tiles[0])
    monkeypatch.setattr(tilemap, "_label_maps", lambda size: official_tiles[1])
    return tilemap._cache_path(2)


def test_atlas_reload_matches_official_and_recoloured_tiles(atlas_cache, monkeypatch):
    built = tilemap.MiniGridAtlas(2, "cpu")
    palette = tilemap.official_colors()
    palette[0] = [12, 34, 56]
    expected = built.rgb({"palette": palette})
    monkeypatch.setattr(tilemap, "_official_atlas", lambda _: pytest.fail("valid cache rebuilt"))
    loaded = tilemap.MiniGridAtlas(2, "cpu")
    assert torch.equal(loaded.official, built.official)
    assert torch.equal(loaded.rgb({"palette": palette}), expected)
    assert list(atlas_cache.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("damage", ["version", "format", "shape", "corrupt"])
def test_stale_or_invalid_atlas_is_rebuilt(atlas_cache, official_tiles, damage):
    tilemap.MiniGridAtlas._load_or_build(2, True)
    with np.load(atlas_cache, allow_pickle=False) as archive:
        data = dict(archive)
    if damage == "version":
        data["minigrid_version"] = "different-version"
    elif damage == "format":
        del data["format"]
    elif damage == "shape":
        data["atlas"] = data["atlas"][:1]
    if damage == "corrupt":
        atlas_cache.write_bytes(b"broken")
    else:
        np.savez(atlas_cache, **data)
    with pytest.warns(RuntimeWarning, match="Rebuilding MiniGrid cache"):
        rgb, labels = tilemap.MiniGridAtlas._load_or_build(2, True)
    np.testing.assert_array_equal(rgb, official_tiles[0])
    np.testing.assert_array_equal(labels, official_tiles[1])


@pytest.mark.parametrize("failure", ["mkdir", "replace"])
def test_unwritable_cache_keeps_rendering_and_removes_temp(atlas_cache, official_tiles, monkeypatch, failure):
    def denied(*args, **kwargs):
        raise PermissionError("read-only cache")

    if failure == "mkdir":
        monkeypatch.setattr(type(atlas_cache), "mkdir", denied)
    else:
        monkeypatch.setattr(tilemap.os, "replace", denied)
    with pytest.warns(RuntimeWarning, match="memory-only"):
        atlas = tilemap.MiniGridAtlas(2, "cpu")
    np.testing.assert_array_equal(atlas.official.numpy(), official_tiles[0])
    assert list(atlas_cache.parent.glob("*.tmp")) == []


def test_concurrent_get_builds_one_atlas(monkeypatch):
    calls = []
    start = threading.Barrier(4)

    def build(self, tile_size, device):
        calls.append(tile_size)
        time.sleep(0.02)

    monkeypatch.setattr(tilemap.MiniGridAtlas, "_mem", {})
    monkeypatch.setattr(tilemap.MiniGridAtlas, "__init__", build)

    def get(_):
        start.wait(timeout=5)
        return tilemap.MiniGridAtlas.get(2, "cpu")

    with ThreadPoolExecutor(max_workers=4) as pool:
        atlases = list(pool.map(get, range(4)))
    assert calls == [2]
    assert all(atlas is atlases[0] for atlas in atlases)
