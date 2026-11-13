"""Rocket frames of the Numba TinyRenderer port (``tiny_raster``) against PyBullet's TinyRenderer.

The port draws the meshes TinyRenderer builds with TinyRenderer's arithmetic, so on the same poses and colours its
frames are PyBullet's frames. The tests report raw per-frame MSE (0-255 scale) and the share of equal frames.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pybullet = pytest.importorskip("pybullet")
pytest.importorskip("numba")
pytest.importorskip("scipy")

import stable_envs as se  # noqa: E402
from stable_envs.envs.rocket import legacy as RL  # noqa: E402
from stable_envs.envs.rocket import render as RR  # noqa: E402
from stable_envs.envs.rocket import tiny_raster as TR  # noqa: E402

#: Mean and worst per-frame MSE of the port against PyBullet's TinyRenderer on the same inputs.
MSE_MEAN_MAX = 0.5
MSE_FRAME_MAX = 5.0


def _inputs(n, seed, colours=False):
    """``(body, links, rgba)`` of ``n`` sampled pose rows: high flight, descent and near-pad poses."""
    rng = np.random.default_rng(seed)
    with se.make("PyFlytRocketLanding", device="cpu", num_envs=1, size=32) as env:
        rows = env.sample_states(n, rng=rng)
        style = env._renderer_rows(np.tile(env.default_style(), (n, 1))).cpu().numpy()
    body, links, rgba = RR.pose_row_fields(rows, style)
    if colours:
        rgba[:, :5, :3] = rng.integers(0, 256, (n, 5, 3)) / 255.0
        rgba[:, RR.FLAME_COLOR] = np.where(rng.random((n, 1)) < 0.5, RL.IGNITION_RGBA, RL.FLAME_RGBA)
    return body, links, rgba


def _pybullet(body, links, rgba, size, ss=1):
    scene = RR.tiny_scene()
    try:
        return RR.draw_tiny(scene, body, links, rgba, size, ss)
    finally:
        pybullet.disconnect(scene[0])


def _mse(a, b):
    return ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean((1, 2, 3))


def _check(ours, theirs, record_property, tag):
    per_frame = _mse(ours, theirs)
    record_property(f"{tag}_mse_mean", float(per_frame.mean()))
    record_property(f"{tag}_mse_max", float(per_frame.max()))
    record_property(f"{tag}_equal_frames", f"{int((per_frame == 0).sum())}/{len(per_frame)}")
    print(f"{tag}: raw MSE mean {per_frame.mean():.4f} max {per_frame.max():.4f}, equal {(per_frame == 0).sum()}")
    assert per_frame.mean() <= MSE_MEAN_MAX and per_frame.max() <= MSE_FRAME_MAX, np.round(per_frame, 3)


@pytest.mark.parametrize("size, ss", [(96, 1), (480, 1), (64, 2)])
def test_frames_equal_pybullet_tinyrenderer(size, ss, record_property):
    body, links, rgba = _inputs(24, seed=size + ss)
    _check(
        TR.draw(body, links, rgba, size, ss), _pybullet(body, links, rgba, size, ss), record_property, f"{size}x{ss}"
    )


def test_colour_leaves_equal_pybullet_tinyrenderer(record_property):
    body, links, rgba = _inputs(24, seed=11, colours=True)
    _check(TR.draw(body, links, rgba, 128, 1), _pybullet(body, links, rgba, 128), record_property, "colours")


def test_states_of_variations_draw_as_pybullet(record_property):
    """Full States after resets with every Variation leaf sampled: the ``numba`` and ``pybullet`` backends agree."""
    with se.make("PyFlytRocketLanding", device="cpu", num_envs=6, size=96, seed=1) as env:
        env.reset(seed=3, options={"variation": ["all"]})
        rng = np.random.default_rng(4)
        frames = {"numba": [], "pybullet": []}
        for _ in range(4):
            env.step(rng.uniform(env.action_low, env.action_high, (6, 7)), frames=None)
            state = env.get_state()
            for backend in frames:
                frames[backend].append(env.render(state, backend=backend).numpy())
    ours, theirs = (np.concatenate(frames[k]) for k in ("numba", "pybullet"))
    _check(ours, theirs, record_property, "variations")


def test_sky_colour_is_not_drawn():
    with se.make("PyFlytRocketLanding", device="cpu", num_envs=2, size=64, backend="numba") as env:
        env.reset(seed=5)
        before = env.render()
        env.reset(seed=5, options={"variation_values": {"environment.sky_color": [10, 200, 30]}})
        assert torch.equal(env.render(), before)


def test_flat_shading_shortcut_equals_per_pixel_shading():
    body, links, rgba = _inputs(16, seed=5, colours=True)
    assert np.array_equal(TR.draw(body, links, rgba, 96, 1, fast=True), TR.draw(body, links, rgba, 96, 1, fast=False))


def test_batch_partition_order_and_threads_do_not_change_frames():
    import numba

    body, links, rgba = _inputs(9, seed=7, colours=True)
    whole = TR.draw(body, links, rgba, 64, 1)
    for i in range(len(body)):
        assert np.array_equal(TR.draw(body[i : i + 1], links[i : i + 1], rgba[i : i + 1], 64, 1)[0], whole[i])
    order = np.arange(len(body))[::-1]
    assert np.array_equal(TR.draw(body[order], links[order], rgba[order], 64, 1), whole[order])
    before = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        assert np.array_equal(TR.draw(body, links, rgba, 64, 1), whole)
    finally:
        numba.set_num_threads(before)


def test_view_matrices_are_pybullets():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(256, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[:8] = [0.0, 0.0, 0.0, 1.0]
    q[8:12] = RL.quat_from_euler(np.array([[0.0, np.pi / 2 - 1e-7, 0.3]] * 4))
    links = np.concatenate([rng.uniform(-200, 200, (256, 3)) + [0, 0, 250], q], -1)
    assert np.array_equal(TR.view_matrices(links), RR.frame_views(links, pybullet))


def test_model_matrices_place_the_links_as_pybullet():
    body, _, _ = _inputs(8, seed=9)
    scene = RR.tiny_scene()
    try:
        client, (_, rocket, _), _ = scene
        for i in range(len(body)):
            pybullet.resetBasePositionAndOrientation(rocket, body[i, 1, :3], body[i, 1, 3:], physicsClientId=client)
            models = TR.model_matrices(body[i : i + 1])[0]
            for link in range(pybullet.getNumJoints(rocket, physicsClientId=client)):
                state = pybullet.getLinkState(rocket, link, computeForwardKinematics=1, physicsClientId=client)
                rotation = np.array(pybullet.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                np.testing.assert_allclose(models[link + 2, :3, 3], state[4], rtol=0, atol=1e-4)
                np.testing.assert_allclose(models[link + 2, :3, :3], rotation, rtol=0, atol=1e-6)
    finally:
        pybullet.disconnect(scene[0])


def test_cylinder_hulls_are_closed_outward_meshes_of_support_points():
    for name, _, _, spec, _ in TR.OBJECTS:
        if spec[0] != TR._CYL:
            continue
        vert, norm, face = TR.bullet_cylinder(spec[1], spec[2])
        points = np.array(TR._cylinder_support_points(spec[1], spec[2]))
        assert len(face) == 2 * len(vert) - 4, name
        assert np.abs(vert[:, None] - points[None]).max(-1).min(1).max() < 1e-12, name
        centroid = vert.mean(0)
        for tri in face:
            a, b, c = vert[tri]
            assert np.dot(np.cross(b - a, c - a), a - centroid) > 0, name
        np.testing.assert_allclose(np.linalg.norm(norm, axis=1), 1.0, rtol=0, atol=1e-6)


def test_live_frames_equal_state_frames():
    with se.make("PyFlytRocketLanding", device="cpu", num_envs=3, size=64, backend="numba", seed=2) as env:
        env.reset(seed=6, options={"variation": ["rocket", "pad"]})
        rng = np.random.default_rng(8)
        actions = torch.as_tensor(rng.uniform(env.action_low, env.action_high, (3, 5, 7)))
        out = env.step(actions, states=slice(None), frames=slice(None))
        for t in range(actions.shape[1]):
            assert torch.equal(out["frames"][:, t], env.render(out["states"][:, t]))
        assert torch.equal(env._render_live(), env.render(env.get_state()))


def test_backends_are_checked():
    with pytest.raises(ValueError, match="backend"):
        se.make("PyFlytRocketLanding", device="cpu", num_envs=1, size=32, backend="opengl")
    with se.make("PyFlytRocketLanding", device="cpu", num_envs=1, size=32, backend="tiny") as env:
        env.reset(seed=0)
        assert torch.equal(env.render(), env.render(backend="pybullet"))
        with pytest.raises(ValueError, match="backend"):
            env.render(backend="nvdiffrast")
