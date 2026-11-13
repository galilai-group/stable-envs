"""Cube/Scene ``backend="raster"`` on CPU, the default: the shared MuJoCo rasterizer against OpenGL."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se

pytest.importorskip("ogbench")
pytest.importorskip("numba")

pytestmark = [pytest.mark.integration, pytest.mark.egl]

FAMILIES = [
    pytest.param("OGBCube", marks=pytest.mark.family("ogbench")),
    pytest.param("OGBScene", marks=pytest.mark.family("ogbench")),
]
# One non-default value of each look leaf (Scene's camera angle range is +-5 degrees).
LOOK = {
    "agent.color": [0.2, 0.8, 0.3],
    "cube.color": [0.1, 0.4, 0.9],
    "floor.color": [[0.6, 0.2, 0.1], [0.1, 0.5, 0.6]],
    "light.intensity": [0.35],
    "camera.angle_delta": [[3.0, -2.0]],
    "cube.size": [0.03],
    "lock_color": [[0.1, 0.9, 0.2], [0.9, 0.8, 0.1]],
}


@pytest.fixture(scope="module")
def software_gl() -> None:
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "glfw"):
        pytest.skip("Requires an OSMesa or GLFW software-GL test environment")


def _state(name, lanes=4):
    env = se.make(name, num_envs=lanes, device="cpu", seed=5)
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(8):
        env.step(env.action_spec.sample(env.B, rng), frames=None)
    return env, env.to_render_state()


def _vary(state, leaf, value):
    varied = state.clone()
    shape = varied["appearance", leaf].shape
    value = torch.as_tensor(value, dtype=varied["appearance", leaf].dtype)
    varied["appearance", leaf] = value.reshape(shape[1:]).expand(shape).clone()
    return varied


def _mse(a, b):
    return ((a.double() - b.double()) ** 2).mean(dim=(1, 2, 3))


@pytest.mark.parametrize("name", FAMILIES)
def test_raster_follows_opengl(name, software_gl):
    env, state = _state(name)
    try:
        expected = env.render(state, backend="opengl")
        actual = env.render(state, backend="raster")
        assert torch.equal(env.render(backend="raster"), actual)
        assert torch.equal(env.render(state), actual)
    finally:
        env.close()
    assert actual.shape == expected.shape and actual.dtype == torch.uint8
    mse = _mse(actual, expected)
    assert float(mse.mean()) < 2.0 and float(mse.max()) < 4.0


@pytest.mark.parametrize("name", FAMILIES)
def test_raster_draws_every_look_leaf(name, software_gl):
    env, state = _state(name, lanes=1)
    leaves = [leaf for leaf in LOOK if leaf != "lock_color" or name == "OGBScene"]
    try:
        base = env.render(state, backend="opengl")
        for leaf in leaves:
            varied = _vary(state, leaf, LOOK[leaf])
            expected = env.render(varied, backend="opengl")
            actual = env.render(varied, backend="raster")
            ignored = float(_mse(base, expected).mean())
            assert float(_mse(actual, expected).mean()) < min(2.0, 0.1 * ignored), leaf
    finally:
        env.close()


@pytest.mark.parametrize("name", FAMILIES)
def test_raster_batches_rows_and_mixed_looks_alike(name, software_gl):
    """A row draws the same frame alone, in a batch and in a batch whose rows have other looks."""
    env, state = _state(name)
    try:
        batch = env.render(state, backend="raster")
        for lane in range(len(state)):
            assert torch.equal(env.render(state[lane : lane + 1], backend="raster"), batch[lane : lane + 1])
        mixed = state.clone()
        mixed[1:2] = _vary(state[1:2], "light.intensity", LOOK["light.intensity"])
        mixed[2:3] = _vary(state[2:3], "camera.angle_delta", LOOK["camera.angle_delta"])
        frames = env.render(mixed, backend="raster")
        assert torch.equal(frames[0], batch[0]) and torch.equal(frames[3], batch[3])
        for lane in (1, 2):
            assert torch.equal(frames[lane : lane + 1], env.render(mixed[lane : lane + 1], backend="raster"))
    finally:
        env.close()


def test_raster_supersample_and_resize(software_gl):
    env, state = _state("OGBCube", lanes=2)
    try:
        for size, ss in ((64, 1), (63, 1), (96, 2)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss)
            actual = env.render(state, backend="raster", size=size, supersample=ss)
            assert actual.shape == (2, size, size, 3)
            assert float(_mse(actual, expected).mean()) < 4.0
    finally:
        env.close()


@pytest.mark.parametrize("name", FAMILIES)
def test_render_semantic_shapes_and_depth(name, software_gl):
    env, _ = _state(name, lanes=2)
    try:
        rows = env._pack_render_state(env.get_state())
        seg, depth = env.render_semantic(rows)
    finally:
        env.close()
    seg, depth = np.asarray(seg), np.asarray(depth)
    assert seg.shape == (2, env.size, env.size, 2) and depth.shape == (2, env.size, env.size)
    hit = seg[..., 0] >= 0
    assert hit.mean() > 0.5
    assert set(np.unique(seg[..., 1][hit]).tolist()) == {5}
    assert float(depth[hit].max()) > 0 and np.all(depth >= 0)
