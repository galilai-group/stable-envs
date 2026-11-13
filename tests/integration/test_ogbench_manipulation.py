"""Cube/Scene end to end on the real OGBench source: build, reset, step, restore and render with software OpenGL."""

import pytest
import torch

import stable_envs as se

pytest.importorskip("ogbench")

pytestmark = pytest.mark.integration


def render_or_skip(env, *args, **kwargs):
    try:
        return env.render(*args, **kwargs)
    except (RuntimeError, ImportError) as error:
        if "software OpenGL" in str(error) or "EGL" in str(error):
            pytest.skip(f"CPU rendering requires Mesa software OpenGL: {error}")
        raise


@pytest.mark.parametrize("cell", ["OGBCube", "OGBScene"])
def test_rollout_restore_and_software_opengl_frames_follow_the_simulation(cell):
    with se.make(cell, num_envs=2, size=64, device="cpu", seed=17) as env:
        env.reset(17)
        saved = env.get_state()
        first = render_or_skip(env, saved)
        assert first.shape == (2, 64, 64, 3) and first.dtype == torch.uint8
        assert not torch.equal(first[0], first[1]), "lanes with different seeds drew the same frame"
        assert torch.equal(first, render_or_skip(env, saved)), "rendering is not a function of the State"

        action = torch.ones(2, 5)
        action[:, 4] = -1
        for _ in range(6):
            env.step(action, frames=None)
        moved = render_or_skip(env)
        assert not torch.equal(moved, first), "stepping did not change the frame"
        rolled = env.get_state()

        env.set_state(saved)
        assert torch.equal(render_or_skip(env), first)
        for _ in range(6):
            env.step(action, frames=None)
        replay = env.get_state()
        assert all(torch.equal(replay[key], value) for key, value in rolled.items(True, True))
        assert torch.equal(render_or_skip(env), moved)
