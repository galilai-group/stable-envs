"""The Cube/Scene Numba CPU rasterizer against MuJoCo's OpenGL frames (backend="opengl")."""

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
# One non-default value of each look leaf.
LOOK = {
    "agent.color": [0.2, 0.8, 0.3],
    "cube.color": [0.1, 0.4, 0.9],
    "floor.color": [[0.6, 0.2, 0.1], [0.1, 0.5, 0.6]],
    "light.intensity": [0.35],
    "camera.angle_delta": [[4.0, -3.0]],
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


def _mse(a, b):
    return ((a.double() - b.double()) ** 2).mean(dim=(1, 2, 3))


@pytest.mark.parametrize("name", FAMILIES)
def test_numba_frames_follow_opengl_frames(name, software_gl):
    env, state = _state(name)
    try:
        expected = env.render(state, backend="opengl")
        actual = env.render(state, backend="numba")
        assert torch.equal(actual, env.render(state, backend="numba"))
        assert torch.equal(env.render(backend="numba"), actual)
    finally:
        env.close()
    assert actual.shape == expected.shape and actual.dtype == torch.uint8
    mse = _mse(actual, expected)
    assert float(mse.mean()) < 2.0 and float(mse.max()) < 4.0


@pytest.mark.parametrize("name", FAMILIES)
def test_numba_draws_every_look_leaf(name, software_gl):
    env, state = _state(name, lanes=1)
    leaves = [leaf for leaf in LOOK if leaf != "lock_color" or name == "OGBScene"]
    # Scene's camera angle range is +-5 degrees.
    look = dict(LOOK, **({"camera.angle_delta": [[3.0, -2.0]]} if name == "OGBScene" else {}))
    try:
        base = env.render(state, backend="opengl")
        for leaf in leaves:
            varied = state.clone()
            shape = varied["appearance", leaf].shape
            value = torch.as_tensor(look[leaf], dtype=varied["appearance", leaf].dtype)
            varied["appearance", leaf] = value.reshape(shape[1:]).expand(shape).clone()
            expected = env.render(varied, backend="opengl")
            actual = env.render(varied, backend="numba")
            ignored = float(_mse(base, expected).mean())
            assert float(_mse(actual, expected).mean()) < min(2.0, 0.1 * ignored), leaf
    finally:
        env.close()


def test_numba_supersample_and_resize(software_gl):
    env, state = _state("OGBCube", lanes=2)
    try:
        for size, ss in ((64, 1), (96, 2)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss)
            actual = env.render(state, backend="numba", size=size, supersample=ss)
            assert actual.shape == (2, size, size, 3)
            assert float(_mse(actual, expected).mean()) < 4.0
        together = env.render(state, backend="numba")
        for lane in range(2):
            torch.testing.assert_close(env.render(state[lane : lane + 1], backend="numba"), together[lane : lane + 1])
    finally:
        env.close()
