"""Options and backend selection of the DM Control MuJoCo Warp renderer, without a GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("dm_control")

from stable_envs._core import environ  # noqa: E402
from stable_envs.envs.dmcontrol import warp_render  # noqa: E402
from stable_envs.envs.dmcontrol.warp_render import Options  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_variables(monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_DMC_WARP_OPTIONS", raising=False)
    monkeypatch.delenv("STABLE_ENVS_DMC_CUDA_RENDER", raising=False)
    environ._parsed.cache_clear()
    yield
    environ._parsed.cache_clear()


def test_defaults_and_overrides():
    assert Options.parse() == Options()
    got = Options.parse({"shadows": "one", "samples": 2, "groups": [1, 0, 1], "bvh": "rebuild", "worlds": 512})
    assert (got.shadows, got.samples, got.groups, got.bvh, got.worlds) == ("one", 2, (0, 1), "rebuild", 512)
    assert Options.parse(got) is got
    assert hash(got) == hash(
        Options.parse({"shadows": "one", "samples": 2, "groups": (0, 1), "bvh": "rebuild", "worlds": 512})
    )


def test_the_variable_sets_defaults_under_the_constructor_options(monkeypatch):
    monkeypatch.setenv("STABLE_ENVS_DMC_WARP_OPTIONS", '{"shadows": "none", "worlds": 1024}')
    environ._parsed.cache_clear()
    got = Options.parse({"worlds": 64})
    assert (got.shadows, got.worlds) == ("none", 64)


@pytest.mark.parametrize(
    "bad",
    [
        {"no_such_option": 1},
        {"shadows": "some"},
        {"bvh": "never"},
        {"samples": 0},
        {"worlds": True},
        {"groups": [5]},
        {"groups": []},
        {"graph": "yes"},
    ],
)
def test_invalid_options_raise(bad):
    with pytest.raises(ValueError):
        Options.parse(bad)


@pytest.mark.parametrize("text", ["[1, 2]", "{bad json"])
def test_an_invalid_variable_names_itself(monkeypatch, text):
    monkeypatch.setenv("STABLE_ENVS_DMC_WARP_OPTIONS", text)
    environ._parsed.cache_clear()
    with pytest.raises(ValueError, match="STABLE_ENVS_DMC_WARP_OPTIONS"):
        Options.parse()


@pytest.mark.parametrize(
    ("backend", "device", "variable", "expected"),
    [
        ("mjwarp", "cuda", None, True),
        ("auto", "cuda", None, False),
        ("auto", "cuda", "mjwarp", True),
        ("auto", "cpu", "mjwarp", False),
        ("numba", "cuda", "mjwarp", False),
        ("opengl", "cuda", "mjwarp", False),
    ],
)
def test_backend_selection(monkeypatch, backend, device, variable, expected):
    if variable is not None:
        monkeypatch.setenv("STABLE_ENVS_DMC_CUDA_RENDER", variable)
        environ._parsed.cache_clear()
    env = SimpleNamespace(backend=backend, device=torch.device(device))
    assert warp_render.selected(env) is expected


def test_the_variable_leaves_manipulator_on_its_renderer(monkeypatch):
    monkeypatch.setenv("STABLE_ENVS_DMC_CUDA_RENDER", "mjwarp")
    environ._parsed.cache_clear()
    env = SimpleNamespace(backend="auto", device=torch.device("cuda"), DOMAIN="manipulator")
    assert warp_render.selected(env) is False


@pytest.mark.parametrize(("backend", "device", "expected"), [("auto", "cuda", True), ("auto", "cpu", False)])
def test_body_tracking_cells_select_it_by_default(backend, device, expected):
    from unittest import mock

    from stable_envs.envs.dmcontrol.trackcom import DMControlTrackCom

    env = mock.MagicMock(spec=DMControlTrackCom)
    env.backend, env.device = backend, torch.device(device)
    assert warp_render.selected(env) is expected


def test_the_render_variable_takes_auto_or_mjwarp(monkeypatch):
    monkeypatch.setenv("STABLE_ENVS_DMC_CUDA_RENDER", "opengl")
    environ._parsed.cache_clear()
    with pytest.raises(ValueError, match="STABLE_ENVS_DMC_CUDA_RENDER"):
        environ.read("STABLE_ENVS_DMC_CUDA_RENDER")
