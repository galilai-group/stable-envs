"""render.mujoco_scene: the geoms, lights, camera and textures that MuJoCo's OpenGL renderer draws."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

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
    <light name="spot" pos="0 0 1" dir="0 0 -1" cutoff="100" castshadow="false"/>
    <camera name="cam" pos="1 0 1" xyaxes="0 1 0 -0.7 0 0.7"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="grid"/>
    <body name="b" pos="0 0 0.2">
      <freejoint/>
      <geom name="box" type="box" size="0.05 0.06 0.07" rgba="1 0 0 1"/>
      <geom name="cyl" type="cylinder" size="0.02 0.03" pos="0.2 0 0" rgba="0 1 0 0.5"/>
      <geom name="cap" type="capsule" size="0.02 0.05" pos="-0.2 0 0" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def scene():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, ms.extract(m, d, "cam")


def test_geoms_and_tessellations(scene):
    m, s = scene
    names = [m.geom(int(g)).name for g in s.objid]
    assert names == ["floor", "box", "cyl", "cap"]
    q = m.vis.quality
    nslice, nstack = int(q.numslices), int(q.numstacks)
    counts = [len(v) for v in s.verts]
    assert counts[1] == 6 * 2 * int(q.numquads) ** 2
    disk = nslice + 2 * nslice * (nstack // 2 - 1)
    assert counts[2] == 2 * disk + 2 * nslice * nstack
    half = nslice + 2 * nslice * (nstack // 2 - 1)
    assert counts[3] == 2 * nslice * nstack + 2 * half
    # Every corner lies within the geom's bounding radius and the box keeps its half sizes.
    for v, rb in zip(s.verts[1:], s.rbound[1:]):
        assert np.linalg.norm(v, axis=-1).max() <= rb * (1 + 1e-5)
    np.testing.assert_allclose(np.abs(s.verts[1]).reshape(-1, 3).max(0), [0.05, 0.06, 0.07], rtol=1e-6)
    assert s.transparent.tolist() == [False, False, True, False]
    assert s.static.tolist() == [True, False, False, False]
    for n in s.normals:
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, rtol=1e-6)


def test_lights_camera_textures(scene):
    m, s = scene
    assert [light["headlight"] for light in s.lights] == [True, False, False]
    assert [light["castshadow"] for light in s.lights] == [False, True, False]
    # GL rejects the 100-degree cutoff, so the spot light keeps 180 degrees.
    assert s.lights[2]["cutoff"] == 180.0
    np.testing.assert_allclose(s.camera["pos"], [1, 0, 1], atol=1e-6)
    assert s.texmode.tolist() == [ms.TEX_2D, ms.TEX_NONE, ms.TEX_NONE, ms.TEX_NONE]
    levels = s.textures[0][0]
    assert [level.shape[0] for level in levels] == [64, 32, 16, 8, 4, 2, 1]
    assert len(s.sky) == 6 and s.sky[0][0].shape == (32, 32, 3)
    win = ms.shadow_matrix(s, s.lights[1])
    look = ms.shadow_lookup(s, s.lights[1])
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    # The light looks down at the origin from 2 m: the map centre, inside the near-far depth range.
    np.testing.assert_allclose((win @ origin)[:2], [s.shadow_size / 2] * 2, atol=1e-6)
    np.testing.assert_allclose((look @ origin)[:2], [s.shadow_size / 2] * 2, atol=1e-6)
    assert 0.0 < (win @ origin)[2] < 1.0
