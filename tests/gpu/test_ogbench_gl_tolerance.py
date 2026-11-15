"""A fresh Cube/Scene env redraws stored states within the documented NVIDIA OpenGL variation."""

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

pytestmark = [pytest.mark.gpu]

#: Documented variation of NVIDIA OpenGL redraws: one level, a few pixels, a few frames.
MAX_LEVEL = 1
MAX_PIXELS = 32
MAX_FRAMES = 0.05


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=pytest.mark.family("ogbench")) for n in ("OGBCube", "OGBScene")]
)
def test_fresh_env_redraws_stored_states_within_the_driver_variation(name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA OpenGL rendering requires a GPU")
    first = se.make(name, num_envs=1, device="cuda")
    try:
        rows = C.sample_render_rows(first, 64, rng=np.random.default_rng(20261023))
        stored = first._render(rows).cpu().numpy().astype(np.int16)
    finally:
        first.close()
    second = se.make(name, num_envs=1, device="cuda")
    try:
        again = second._render(rows).cpu().numpy().astype(np.int16)
    finally:
        second.close()
    diff = np.abs(again - stored)
    pixels = (diff > 0).any(-1).reshape(len(diff), -1).sum(1)
    assert int(diff.max()) <= MAX_LEVEL
    assert int(pixels.max()) <= MAX_PIXELS
    assert int((pixels > 0).sum()) <= MAX_FRAMES * len(diff)


#: Largest mean squared 0-255 difference between CUDA frames drawn by the OpenGL workers and by one process. Each
#: worker has its own context, so only the driver variation above remains. RTX A5000 and A5500 hosts measured below 1e-4 for Cube and
#: 1.3e-3 for Scene at 8 states.
WORKER_MSE = 2e-3


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=pytest.mark.family("ogbench")) for n in ("OGBCube", "OGBScene")]
)
def test_worker_frames_stay_within_the_measured_mse_of_one_process(name):
    from stable_envs.envs.ogbench import manipulation_opengl_pool as pool

    if not torch.cuda.is_available():
        pytest.skip("CUDA OpenGL rendering requires a GPU")
    env = se.make(name, num_envs=1, device="cuda")
    try:
        rows = C.sample_render_rows(env, 8, rng=np.random.default_rng(20261115))
        if pool.workers(len(rows), "cuda") < 2:
            pytest.skip("OpenGL workers need Linux and at least two CPUs")
        workers = env._render(rows).cpu().numpy().astype(np.float64)
        with se.tuning.override(ogb_gl_workers=0):
            alone = env._render(rows).cpu().numpy().astype(np.float64)
    finally:
        env.close()
    mse = float(((workers - alone) ** 2).mean())
    assert mse <= WORKER_MSE, f"{name}: worker vs one-process frames MSE {mse:.3g}"
    assert int(np.abs(workers - alone).max()) <= MAX_LEVEL
