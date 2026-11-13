"""CUDA Rocket frames against the CPU TinyRenderer port on the same poses and colours.

The CUDA path uses TinyRenderer's meshes, matrices, degenerate-triangle rule and colour law; its coverage and normal
interpolation are float32. The test reports raw per-frame MSE (0-255 scale).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.regression]

if not torch.cuda.is_available():
    pytest.skip("Rocket CUDA frames require a CUDA device", allow_module_level=True)

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")
pytest.importorskip("nvdiffrast")
pytest.importorskip("numba")

import stable_envs as se  # noqa: E402
from stable_envs.envs.rocket import legacy as RL  # noqa: E402
from stable_envs.envs.rocket import render as RR  # noqa: E402
from stable_envs.envs.rocket import tiny_raster as TR  # noqa: E402

SIZE, N = 480, 32
MSE_MEAN_MAX = 5.0
MSE_FRAME_MAX = 25.0


@pytest.mark.parametrize("colours", [False, True])
def test_cuda_frames_track_the_tinyrenderer_port(colours, record_property):
    rng = np.random.default_rng(21 + colours)
    with se.make("PyFlytRocketLanding", num_envs=N, size=SIZE, device="cuda") as env:
        rows = env.sample_states(N, rng=rng)
        style = env._renderer_rows(np.tile(env.default_style(), (N, 1))).cpu().numpy()
        body, links, rgba = RR.pose_row_fields(rows, style)
        if colours:
            rgba[:, :5, :3] = rng.integers(0, 256, (N, 5, 3)) / 255.0
            rgba[:, RR.FLAME_COLOR] = np.where(rng.random((N, 1)) < 0.5, RL.IGNITION_RGBA, RL.FLAME_RGBA)
        ours = RR.draw_gpu(env, body, links, rgba, SIZE, 1).cpu().numpy().astype(np.float64)
        rows_frames = env.render(torch.as_tensor(rows, device="cuda"), style=torch.as_tensor(style)).cpu().numpy()
    port = TR.draw(body, links, rgba, SIZE, 1).astype(np.float64)
    per_frame = ((ours - port) ** 2).mean((1, 2, 3))
    record_property("mse_mean", float(per_frame.mean()))
    record_property("mse_max", float(per_frame.max()))
    print(f"CUDA vs port, colours={colours}: raw MSE mean {per_frame.mean():.4f} max {per_frame.max():.4f}")
    assert per_frame.mean() <= MSE_MEAN_MAX and per_frame.max() <= MSE_FRAME_MAX, np.round(per_frame, 2)
    if not colours:
        assert np.array_equal(rows_frames, ours.astype(np.uint8))
