"""Fetch ``backend="raster"`` on CPU, the default: the shared MuJoCo rasterizer against MuJoCo OpenGL."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se

pytestmark = pytest.mark.egl

SCENES = ["Reach", "Push", "Slide", "PickAndPlace"]
VISUAL = ["table.color", "background.color", "light.intensity", "camera.angle_delta", "rendering.transparent_arm"]
#: One non-default value of each Fetch look leaf.
LOOK = {
    "floor.color": [0.6, 0.3, 0.2],
    "table.color": [0.2, 0.5, 0.7],
    "object.color": [0.1, 0.8, 0.3],
    "arm.color": [0.8, 0.2, 0.2],
    "goal.color": [0.2, 0.2, 0.9],
    "sky.color": [0.3, 0.3, 0.6],
    "rendering.transparent_arm": [1],
    "light.diffuse": [0.5],
    "camera.distance_scale": [0.8],
    "camera.azimuth_delta": [15.0],
    "camera.elevation_delta": [-10.0],
}


@pytest.fixture(scope="module")
def software_gl() -> None:
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "glfw"):
        pytest.skip("Requires an OSMesa or GLFW software-GL test environment")
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("numba")


def _mse(a, b):
    return ((a.double() - b.double()) ** 2).reshape(len(a), -1).mean(1)


def _env(scene, lanes, varied=False):
    env = se.make(f"Fetch{scene}", num_envs=lanes, device="cpu")
    options = {"variation": VISUAL + ([] if scene == "Reach" else ["object.color"])} if varied else None
    env.reset(seed=3, options=options)
    generator = torch.Generator().manual_seed(3)
    for _ in range(4):
        env.step(torch.rand(lanes, 4, generator=generator) * 2 - 1)
    return env


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("scene", SCENES)
def test_raster_follows_opengl(scene: str, varied: bool, software_gl: None) -> None:
    with _env(scene, 4, varied) as env:
        state = env.get_state()
        expected = env.render(state, backend="opengl")
        actual = env.render(state, backend="raster")
        assert torch.equal(env.render(state), actual)
    mse = _mse(actual, expected)
    assert float(mse.max()) < 0.5


@pytest.mark.parametrize("scene", ["Reach", "PickAndPlace"])
def test_raster_draws_every_look_leaf(scene: str, software_gl: None) -> None:
    with _env(scene, 1) as env:
        rows = env._pack_render_state(env.get_state())
        base = env._render(rows, backend="opengl")
        for leaf, value in LOOK.items():
            if leaf == "object.color" and scene == "Reach":
                continue
            variation = {leaf: np.asarray(value, np.float32)}
            expected = env._render(rows, variation=variation, backend="opengl")
            actual = env._render(rows, variation=variation, backend="raster")
            ignored = float(_mse(base, expected).mean())
            assert ignored > 0, leaf
            assert float(_mse(actual, expected).mean()) < min(0.5, 0.1 * ignored), leaf


def test_raster_rows_batches_sizes(software_gl: None) -> None:
    with _env("Push", 3, varied=True) as env:
        state = env.get_state()
        rows = env._pack_render_state(state)
        together = env._render(rows, backend="raster")
        for lane in range(3):
            assert torch.equal(env._render(rows[lane : lane + 1], backend="raster"), together[lane : lane + 1])
        for size, ss in ((1, 1), (63, 1), (96, 2), (64, 3)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss).double()
            actual = env.render(state, backend="raster", size=size, supersample=ss).double()
            assert actual.shape == (3, size, size, 3)
            assert float(((actual - expected) ** 2).mean()) < (2.0 if size > 8 else 40.0)
        out = torch.empty((3, 64, 64, 3), dtype=torch.uint8)
        returned = env.render(state, size=64, out=out, backend="raster")
        assert returned.data_ptr() == out.data_ptr()


def test_raster_clips_triangles_at_the_near_plane(software_gl: None) -> None:
    """A camera close to the table cuts its triangles at the near plane instead of dropping them."""
    with _env("Push", 1) as env:
        rows = env._pack_render_state(env.get_state())
        variation = {"camera.distance_scale": np.asarray([0.6], np.float32)}
        expected = env._render(rows, variation=variation, backend="opengl")
        actual = env._render(rows, variation=variation, backend="raster")
    assert float(_mse(actual, expected).max()) < 1.0


def test_render_semantic_maps_onto_the_reference_model(software_gl: None) -> None:
    import mujoco

    from stable_envs.testing.conformance import reference_env

    with _env("Push", 2) as env:
        rows = env._pack_render_state(env.get_state())
        seg, depth = env.render_semantic(rows)
        ref = reference_env(type(env), env.size).unwrapped.model
    seg, depth = np.asarray(seg), np.asarray(depth)
    assert seg.shape == (2, env.size, env.size, 2) and depth.shape == (2, env.size, env.size)
    hit = seg[..., 0] >= 0
    assert hit.mean() > 0.5
    geoms = np.unique(seg[..., 0][hit & (seg[..., 1] == 5)])
    names = {mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in geoms}
    assert {"floor0", "object0"} <= names
    assert np.all(depth >= 0) and float(depth[hit].max()) > 0
