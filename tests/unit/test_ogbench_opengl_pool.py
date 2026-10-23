"""OpenGL worker processes return the frames the main process draws, in row order."""

import os
import sys

import pytest
import torch

import stable_envs as se
from stable_envs import tuning


def same(frames, reference, device):
    """Software OpenGL frames are equal; NVIDIA OpenGL redraws can differ in a few pixels of a rare frame."""
    if device == "cpu":
        assert torch.equal(frames, reference)
        return
    pixels = (frames != reference).any(-1).flatten(1).sum(1)
    assert int((pixels > 0).sum()) <= 2 and int(pixels.max()) <= 64


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("OGBCube", marks=pytest.mark.family("ogbench")),
        pytest.param("OGBScene", marks=pytest.mark.family("ogbench")),
    ],
)
@pytest.mark.egl
def test_worker_frames_equal_in_process_frames(name):
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenGL workers run on Linux only")
    selection = os.environ.get("SE_FIDELITY_TEST_FAMILY", "")
    if selection and not name.startswith(selection):
        pytest.skip("Different pinned family profile")
    from stable_envs.envs.ogbench import manipulation_opengl_pool as pool

    device = os.environ.get("SE_FIDELITY_TEST_DEVICE", "cpu")
    env = se.make(name, num_envs=24, device=device, seed=17, backend="opengl")
    try:
        env.reset(seed=3)
        state = env.to_render_state()
        if name == "OGBScene":
            state["pose", "buttons"][::2] = 1 - state["pose", "buttons"][::2]
        with tuning.override(ogb_gl_workers=0):
            assert pool.workers(len(state)) == 0
            alone = env.render(state, backend="opengl")
        with tuning.override(ogb_gl_workers=3):
            if pool.workers(len(state)) == 0:
                pytest.skip("Fewer than 2 CPUs in the process affinity")
            shared = env.render(state, backend="opengl")
            again = env.render(state[3:19], backend="opengl")
        assert shared.device == alone.device and shared.dtype == torch.uint8
        same(shared, alone, device)
        same(again, alone[3:19], device)
    finally:
        env.close()
