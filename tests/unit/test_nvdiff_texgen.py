"""Guard the constants in ``stable_envs.render.nvdiff`` that were MEASURED against MuJoCo.

Every one of these was wrong on a first reading and right only after a frame said so
(``pre-repository-cleanup:docs/research/nvdiffrast.md`` Section 2). They are cheap to state and expensive to rediscover, so
they are asserted here rather than left to the GPU parity run: this file needs numpy and ``mujoco``
and no CUDA, no nvdiffrast and no reference env.

The texgen law:

    u = +0.5 * texrepeat_x * x * scale_x + 0.5
    v = -0.5 * texrepeat_y * y * scale_y + 0.5
    scale = 1 (texuniform) else 1 / half_extent

with ``half_extent`` the per-axis ``geom_size`` for box / ellipsoid / plane, ``(r, r, r)`` for a
sphere, and the RADIUS on both texture axes for a capsule and a cylinder.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stable_envs.render import nvdiff as ND  # noqa: E402

XML = """
<mujoco>
  <asset>
    <texture name="t" type="{ttype}" builtin="flat" width="64" height="64" rgb1="1 1 1"/>
    <material name="m" texture="t" texrepeat="{rx} {ry}" texuniform="{uni}"/>
  </asset>
  <worldbody>
    <geom name="g" type="{gtype}" size="{size}" material="m"/>
  </worldbody>
</mujoco>
"""


def _geom(gtype, size, rx=1.0, ry=1.0, uni=False, ttype="2d"):
    m = mujoco.MjModel.from_xml_string(
        XML.format(gtype=gtype, size=size, rx=rx, ry=ry, uni="true" if uni else "false", ttype=ttype)
    )
    return m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g")


@pytest.mark.parametrize(
    "gtype,size,rep,uni,du_dx,dv_dy",
    [
        # texuniform=false: the repeat spans the OBJECT, so the slope is 0.5*repeat/half_extent
        ("box", "0.4 0.3 0.2", 0.5, False, 0.5 * 0.5 / 0.4, -0.5 * 0.5 / 0.3),
        ("sphere", "0.4", 0.5, False, 0.5 * 0.5 / 0.4, -0.5 * 0.5 / 0.4),
        # a capsule and a cylinder use their RADIUS on both axes, not their half-length
        ("capsule", "0.3 0.4", 0.5, False, 0.5 * 0.5 / 0.3, -0.5 * 0.5 / 0.3),
        ("cylinder", "0.4 0.6", 0.5, False, 0.5 * 0.5 / 0.4, -0.5 * 0.5 / 0.4),
        # a plane is NOT special: same law, same size term. mujoco_warp's rule omits the size term,
        # which is right only when texuniform is true.
        ("plane", "16 16 0.2", 1.0, False, 0.5 / 16.0, -0.5 / 16.0),
        ("plane", "0.5 0.5 0.1", 1.0, False, 0.5 / 0.5, -0.5 / 0.5),
        # texuniform=true: the repeat is per unit LENGTH, so the size term disappears entirely
        ("box", "0.4 0.3 0.2", 1.0, True, 0.5, -0.5),
        ("plane", "16 16 0.2", 1.0, True, 0.5, -0.5),
        ("sphere", "0.4", 2.0, True, 1.0, -1.0),
    ],
)
def test_texgen_matches_the_measured_law(gtype, size, rep, uni, du_dx, dv_dy):
    m, g = _geom(gtype, size, rx=rep, ry=rep, uni=uni)
    A, b = ND.texgen(m, g)
    assert A[0, 0] == pytest.approx(du_dx, rel=1e-6)
    assert A[1, 1] == pytest.approx(dv_dy, rel=1e-6)
    assert b[0] == pytest.approx(0.5) and b[1] == pytest.approx(0.5)
    assert A[0, 1] == 0.0 and A[1, 0] == 0.0, "u depends on local x alone and v on local y alone"


def test_cube_texture_is_a_direction_not_a_uv():
    """A cube map is indexed by a direction; the 0.5 offset that centres a 2D map would tilt it."""
    m, g = _geom("sphere", "0.4", ttype="cube")
    A, b = ND.texgen(m, g, cube=True)
    assert np.allclose(A, np.eye(3)) and np.allclose(b, 0.0)


def test_effective_rgba_prefers_the_geoms_own_colour():
    """MuJoCo's resolution order, which MJWarp inverts: the material only wins on the DEFAULT rgba."""
    m = mujoco.MjModel.from_xml_string("""
    <mujoco><asset><material name="m" rgba="0.1 0.2 0.3 0.4"/></asset>
    <worldbody>
      <geom name="own" type="box" size="1 1 1" material="m" rgba="0.9 0.8 0.7 0.6"/>
      <geom name="def" type="box" size="1 1 1" material="m"/>
    </worldbody></mujoco>""")
    own = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "own")
    dfl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "def")
    assert np.allclose(ND.effective_rgba(m, own), [0.9, 0.8, 0.7, 0.6])
    assert np.allclose(ND.effective_rgba(m, dfl), [0.1, 0.2, 0.3, 0.4])


def test_pow2_resample_is_lossless_on_a_two_by_two_checker():
    """Why the locomotion floor survives the resample nvdiffrast forces, stated as a test.

    nvdiffrast will not build a pyramid over an odd extent, so Gymnasium's 100x100 checker has to be
    resampled to 128x128 before it can be minified at all. That is a change to the picture in general
    and NOT a change to this picture: every 50-texel source block maps onto a whole 64-texel
    destination block, so the resampled texture is the same 2x2 checker. OGBench's 300x300 per-texel
    gradient is the case where it does cost something, and that cost is measured in
    pre-repository-cleanup:docs/research/nvdiffrast.md Section 3, not asserted here.
    """
    a = np.zeros((100, 100, 3), np.float32)
    a[:50, :50] = a[50:, 50:] = 1.0
    out, note = ND._to_pow2(a)
    assert out.shape == (128, 128, 3) and note == "100x100 -> 128x128 (nearest)"
    want = np.zeros((128, 128, 3), np.float32)
    want[:64, :64] = want[64:, 64:] = 1.0
    assert np.array_equal(out, want)


def test_quantisation_rounds_rather_than_truncates():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.0, 0.5 / 255, 1.4 / 255, 1.6 / 255, 1.0, 2.0, -1.0]])
    assert ND.QUANT(x, "round").tolist() == [[0, 0, 1, 2, 255, 255, 0]]
    assert ND.QUANT(x, "trunc").tolist() == [[0, 0, 1, 1, 255, 255, 0]]
    with pytest.raises(ValueError):
        ND.QUANT(x, "nearest-even")
