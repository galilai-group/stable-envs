import os

import numpy as np
import pytest
import torch

import stable_envs as se

pytestmark = pytest.mark.egl

VISUAL = ["table.color", "background.color", "light.intensity", "camera.angle_delta", "rendering.transparent_arm"]


@pytest.fixture(scope="module")
def software_gl() -> None:
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "glfw"):
        pytest.skip("Requires an OSMesa or GLFW software-GL test environment")
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("numba")


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("scene", ["Reach", "Push", "Slide", "PickAndPlace"])
def test_numba_frames_follow_opengl_frames(scene: str, varied: bool, software_gl: None) -> None:
    with se.make(f"Fetch{scene}", num_envs=4, device="cpu") as env:
        options = {"variation": VISUAL + ([] if scene == "Reach" else ["object.color"])} if varied else None
        env.reset(seed=3, options=options)
        generator = torch.Generator().manual_seed(3)
        for _ in range(4):
            env.step(torch.rand(4, 4, generator=generator) * 2 - 1)
        state = env.get_state()
        expected = env.render(state, backend="opengl").numpy().astype(np.float64)
        actual = env.render(state).numpy().astype(np.float64)
        np.testing.assert_array_equal(actual, env.render(state, backend="numba").numpy())
    mse = ((actual - expected) ** 2).reshape(4, -1).mean(1)
    assert mse.max() < 0.5
    assert np.abs(actual - expected).mean() < 0.05


def test_numba_supersample_and_resize(software_gl: None) -> None:
    with se.make("FetchPush", num_envs=2, device="cpu") as env:
        env.reset(seed=5)
        state = env.get_state()
        for size, ss in ((1, 1), (63, 1), (96, 2), (64, 3)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss).numpy().astype(np.float64)
            actual = env.render(state, backend="numba", size=size, supersample=ss).numpy().astype(np.float64)
            assert actual.shape == (2, size, size, 3)
            assert ((actual - expected) ** 2).mean() < (2.0 if size > 8 else 40.0)
        rows = env._pack_render_state(state)
        together = env._render(rows, backend="numba")
        for lane in range(2):
            torch.testing.assert_close(env._render(rows[lane : lane + 1], backend="numba"), together[lane : lane + 1])
        out = torch.empty((2, 64, 64, 3), dtype=torch.uint8)
        returned = env.render(state, size=64, out=out)
        assert returned.data_ptr() == out.data_ptr()
