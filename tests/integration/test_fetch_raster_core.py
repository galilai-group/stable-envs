"""Fetch ``render_semantic`` on CPU: the shared MuJoCo rasterizer's segmentation and depth against the frame."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se

pytestmark = pytest.mark.egl


@pytest.fixture(scope="module")
def software_gl() -> None:
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "glfw"):
        pytest.skip("Requires an OSMesa or GLFW software-GL test environment")
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("numba")


def _env(scene, lanes):
    env = se.make(f"Fetch{scene}", num_envs=lanes, device="cpu")
    env.reset(seed=3)
    generator = torch.Generator().manual_seed(3)
    for _ in range(4):
        env.step(torch.rand(lanes, 4, generator=generator) * 2 - 1)
    return env


def test_render_semantic_maps_onto_the_reference_model(software_gl: None) -> None:
    import mujoco

    from stable_envs.testing.conformance import reference_env

    with _env("Push", 2) as env:
        rows = env._pack_render_state(env.get_state())
        seg, depth = env.render_semantic(rows)
        ref = reference_env(type(env), env.size).unwrapped.model
        # Pixels whose colour follows sky.color show the sky; every other pixel shows a geom.
        skies = [
            env._render(rows, variation={"sky.color": np.asarray(c, np.float32)}, backend="numba").numpy()
            for c in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        ]
    seg, depth = np.asarray(seg), np.asarray(depth)
    assert seg.shape == (2, env.size, env.size, 2) and depth.shape == (2, env.size, env.size)
    hit = seg[..., 0] >= 0
    sky = np.any(skies[0] != skies[1], axis=-1)
    # Antialiased silhouette pixels mix sky and geom colour, so a few disagree at the edges.
    assert np.mean(hit == ~sky) > 0.98
    assert 0.05 < hit.mean() < 0.95
    geoms = np.unique(seg[..., 0][hit & (seg[..., 1] == 5)])
    names = {mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in geoms}
    assert {"floor0", "object0"} <= names
    assert np.all(depth >= 0) and float(depth[hit].max()) > 0
