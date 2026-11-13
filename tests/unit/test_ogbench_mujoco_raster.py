"""render.mujoco_raster on a small scene: packing, per-frame inputs, plates, clipping and semantics."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("numba")

from stable_envs.render import mujoco_raster as core  # noqa: E402
from stable_envs.render import mujoco_scene as ms  # noqa: E402

XML = """
<mujoco>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".2 .2 .2" width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.2 0.3 0.4" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 2" directional="true"/>
    <camera name="cam" pos="1 0 1" xyaxes="0 1 0 -0.7 0 0.7"/>
    <camera name="down" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
    <camera name="low" pos="-2 0 0.05" xyaxes="0 -1 0 0 0 1"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
    <body name="b" pos="0 0 0.25">
      <freejoint/>
      <geom name="box" type="box" size="0.3 0.3 0.25" rgba="1 0 0 1"/>
      <geom name="cyl" type="cylinder" size="0.02 0.03" pos="0.4 0 0" rgba="0 1 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _scene(camera="cam"):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, ms.pack(ms.extract(m, d, camera))


def _inputs(m, d, p, n):
    gpos = np.repeat(np.asarray(d.geom_xpos[p.objid], np.float64)[None], n, 0)
    gmat = np.repeat(np.asarray(d.geom_xmat[p.objid], np.float64)[None], n, 0)
    rgba = np.repeat(p.rgba[None], n, 0)
    cams = np.repeat(p.cam[None], n, 0)
    lights = ms.headlight_rows(p.lights, cams)
    sky = np.full((n, 3), -1.0)
    return gpos, gmat, rgba, cams, lights, sky


def test_pack_rows_and_headlight():
    _, _, p = _scene()
    assert p.lights.shape[1] == ms.LIGHT_ROW
    assert p.lights[0, 22] == 1.0 and p.lights[1:, 22].tolist() == [0.0] * (len(p.lights) - 1)
    assert p.shadow_light == 1 and p.lights[1, 1] == 1.0
    assert np.all(np.diff(p.cstarts) > 0) and np.all(np.diff(p.starts) > 0)
    assert p.tris.max() < len(p.cpos)
    np.testing.assert_allclose(np.linalg.norm(p.cam[9:12]), 1.0)
    cams = np.repeat(p.cam[None], 2, 0)
    cams[1, 9:12] = [0.0, 0.0, -1.0]
    rows = ms.headlight_rows(p.lights, cams)
    np.testing.assert_allclose(rows[1, 0, 2:5], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rows[0, 0, 2:5], p.lights[0, 2:5], atol=1e-6)
    np.testing.assert_array_equal(rows[:, 1:], np.repeat(p.lights[None, 1:], 2, 0))


def test_transparent_order_ranks_translucent_geoms_only():
    gpos = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])
    cams = np.zeros((1, 12))
    alpha = np.array([[1.0, 0.5, 0.5]])
    rank = ms.transparent_order(gpos, cams, np.zeros(3, np.float32), alpha)
    assert rank.tolist() == [[-1, 1, 0]]


def test_unverified_mujoco_version_is_refused():
    _, _, p = _scene()
    s = ms.Scene()
    s.version = "9.9"
    with pytest.raises(NotImplementedError, match="9.9"):
        ms.require(s, "mesh")
    assert ms.mujoco_version() in ms.VERIFIED


def test_frames_alone_and_in_a_batch_are_equal_and_plates_group_by_look():
    m, d, p = _scene()
    inputs = _inputs(m, d, p, 3)
    gpos, gmat, rgba, cams, lights, sky = inputs
    lights[1, 1, 8:11] *= 0.5
    sky[2] = [10.0, 20.0, 30.0]
    keys, first, index = core.plate_keys(p, rgba, cams, lights, sky)
    assert len(keys) == 3 and len(set(index.tolist())) == 3
    images = np.empty((3, 32, 32, 3), np.uint8)
    core.draw(p, *inputs, 1, images)
    for i in range(3):
        alone = np.empty((1, 32, 32, 3), np.uint8)
        core.draw(p, *(a[i : i + 1] for a in inputs), 1, alone)
        np.testing.assert_array_equal(alone[0], images[i])
    assert not np.array_equal(images[0], images[1])
    again = np.empty_like(images)
    core.draw(p, *inputs, 1, again)
    np.testing.assert_array_equal(again, images)


def test_flat_sky_fills_the_background():
    m, d, p = _scene()
    gpos, gmat, rgba, cams, lights, sky = _inputs(m, d, p, 1)
    sky[0] = [10.0, 20.0, 30.0]
    rgba[0, :, 3] = 0.0
    images = np.empty((1, 16, 16, 3), np.uint8)
    core.draw(p, gpos, gmat, rgba, cams, lights, sky, 1, images)
    assert np.all(images[0] == [10, 20, 30])


def test_semantic_depth_is_the_distance_along_the_optical_axis():
    """Looking straight down from 2 m at the 0.5 m high box top, the box pixels are 1.5 m deep."""
    m, d, p = _scene("down")
    gpos, gmat, rgba, cams, _, _ = _inputs(m, d, p, 1)
    seg, depth = core.semantic(p, gpos, gmat, rgba, cams, 33)
    box = int(np.flatnonzero(p.objid == m.geom("box").id)[0])
    centre = seg[0, 16, 16]
    assert centre == box
    np.testing.assert_allclose(depth[0][seg[0] == box], 1.5, rtol=1e-6)
    floor = int(np.flatnonzero(p.objid == m.geom("floor").id)[0])
    np.testing.assert_allclose(depth[0][seg[0] == floor], 2.0, rtol=1e-6)


def test_triangles_through_the_near_plane_are_clipped():
    """A low camera sees the floor up to its feet: the bottom rows are floor, not skybox."""
    m, d, p = _scene("low")
    inputs = list(_inputs(m, d, p, 1))
    inputs[5][0] = [255.0, 0.0, 255.0]
    images = np.empty((1, 32, 32, 3), np.uint8)
    core.draw(p, *inputs, 1, images)
    bottom = images[0, -1]
    assert not np.any(np.all(bottom == [255, 0, 255], -1))
