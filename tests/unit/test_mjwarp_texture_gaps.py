"""``texture_gaps`` must be a safe all-clear, not a geom-type check wearing one.

WHY THIS FILE EXISTS. ``stable_envs.render.mjwarp.texture_gaps`` answers one question for every
MJWarp-backed family: "are this scene's textures ones MJWarp can sample?" Until Step 7 it answered
it by geom TYPE alone — ``PLANE`` and ``MESH`` are fine, everything else samples texel (0,0) — and
never looked at what KIND of texture was involved.

That is a false all-clear, and it does not fail quietly. ``swm/OGBScene-v0`` returned an EMPTY gap
list and then killed the renderer with ``Warp CUDA error 700: an illegal memory access``, raised
from inside the warm-up frame with no mention of textures in the traceback. The cause, bisected on
an RTX 3090 (``examples/results/step7/ogb-manipulation/oscar/mjwarp/mjwarp_bisect_OGBScene.json``):
its ``btn_metal`` material wears a 512x512 **CUBE** map on two button MESH geoms. MJWarp builds
``wp.Texture2D`` and samples with 2-D coordinates, so a cube map is not drawn badly — it is indexed
out of bounds. Disabling the graph capture, the skybox or shadows all still crashed; ``textures=False``
rendered; and detaching that ONE cube map, leaving every 2-D texture and the skybox in place,
rendered.

The tests are built on SYNTHETIC models, one per rule, because a rule is only proved by a model that
breaks it and the repository's own registered families are exactly the ones that must NOT change.
The last test is the regression tripwire the fix is worth: a cube map on a MESH — the geom type the
old rule declared safe.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stable_envs.render.mjwarp import TEXTURE_SAMPLER_OK, TEXTURED_OK, texture_gaps


def _model(tex_type: str, geom: str, *, textured: bool = True, group: int = 1) -> "mujoco.MjModel":
    """One geom of type ``geom``, optionally wearing a material with a ``tex_type`` texture."""
    if textured:
        if tex_type == "skybox":
            tex = '<texture name="t" type="skybox" builtin="gradient" width="8" height="48"/>'
        elif tex_type == "cube":
            tex = '<texture name="t" type="cube" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="48"/>'
        else:
            tex = '<texture name="t" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        asset = f'<asset>{tex}<material name="m" texture="t"/></asset>'
        matattr = 'material="m"'
    else:
        asset = '<asset><material name="m" rgba="1 0 0 1"/></asset>'
        matattr = 'material="m"'
    size = {"plane": "1 1 .1", "mesh": "", "box": ".1 .1 .1", "sphere": ".1",
            "cylinder": ".1 .1", "capsule": ".1 .1"}[geom]
    if geom == "mesh":
        asset = asset.replace("</asset>",
                              '<mesh name="mm" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>')
        g = f'<geom type="mesh" mesh="mm" group="{group}" {matattr}/>'
    else:
        g = f'<geom type="{geom}" size="{size}" group="{group}" {matattr}/>'
    return mujoco.MjModel.from_xml_string(f"<mujoco>{asset}<worldbody>{g}</worldbody></mujoco>")


def test_constants_say_what_they_mean():
    assert TEXTURED_OK == (0, 7), "PLANE and MESH are the geom types MJWarp can UV"
    assert TEXTURE_SAMPLER_OK == (0,), "mjTEXTURE_2D is the only kind wp.Texture2D can address"
    assert int(mujoco.mjtTexture.mjTEXTURE_2D) == 0
    assert int(mujoco.mjtTexture.mjTEXTURE_CUBE) == 1


@pytest.mark.parametrize("geom", ["plane", "mesh"])
def test_2d_texture_on_a_uv_able_geom_is_not_a_gap(geom):
    assert texture_gaps(_model("2d", geom)) == []


@pytest.mark.parametrize("geom", ["box", "sphere", "cylinder", "capsule"])
def test_2d_texture_on_a_primitive_is_a_gap(geom):
    """Condition 1, unchanged: MJWarp cannot derive UVs, so the surface renders as one texel."""
    gaps = texture_gaps(_model("2d", geom))
    assert len(gaps) == 1
    assert gaps[0][2] == int(getattr(mujoco.mjtGeom, f"mjGEOM_{geom.upper()}"))


@pytest.mark.parametrize("geom", ["plane", "mesh", "box", "sphere"])
def test_an_untextured_material_is_never_a_gap(geom):
    assert texture_gaps(_model("2d", geom, textured=False)) == []


def test_geoms_outside_the_rendered_groups_are_ignored():
    assert texture_gaps(_model("2d", "box", group=1), groups=(0,)) == []
    assert texture_gaps(_model("cube", "mesh", group=1), groups=(0,)) == []


@pytest.mark.parametrize("geom", ["mesh", "plane"])
def test_a_cube_map_on_a_uv_able_geom_IS_a_gap(geom):
    """The regression this fix exists for.

    A cube map on a MESH is the exact shape of ``swm/OGBScene-v0``'s crash, and it is the case the
    geom-type rule declared safe: MESH is in ``TEXTURED_OK``, so the old check returned an empty list
    and the renderer then died with an illegal memory access.
    """
    gaps = texture_gaps(_model("cube", geom))
    assert len(gaps) == 1, "a cube map worn by a geom must be reported whatever the geom type"
    assert gaps[0][2] == int(getattr(mujoco.mjtGeom, f"mjGEOM_{geom.upper()}"))


def test_a_cube_map_on_a_primitive_is_still_a_gap():
    assert len(texture_gaps(_model("cube", "box"))) == 1


def test_a_skybox_material_worn_by_a_geom_is_a_gap():
    """A skybox texture NOT attached to any geom is invisible to this function by construction.

    The function only ever looks at ``geom_matid``, so a model's own skybox — which no geom wears —
    cannot be flagged. This asserts the other half: if a geom actually wears one, it is reported,
    because ``wp.Texture2D`` cannot address it any better than a cube map.
    """
    assert len(texture_gaps(_model("skybox", "mesh"))) == 1


def test_the_models_own_skybox_is_not_flagged():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><asset>'
        '<texture name="sky" type="skybox" builtin="gradient" width="8" height="48"/>'
        '<texture name="t" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        '<material name="m" texture="t"/></asset>'
        '<worldbody><geom type="plane" size="1 1 .1" group="1" material="m"/></worldbody></mujoco>')
    assert int(m.ntex) == 2 and any(int(m.tex_type[i]) == 2 for i in range(m.ntex))
    assert texture_gaps(m) == [], "a skybox no geom wears is not a geom's texture gap"


def test_every_texture_role_is_examined_not_only_the_largest_id():
    """A material can wear a 2-D albedo and a cube environment map; only the second is fatal.

    The previous implementation reduced ``mat_texid`` with ``np.max``, which picks one id by numeric
    accident. Asserted directly on the array so the test fails if that reduction comes back.
    """
    m = _model("2d", "mesh")
    assert texture_gaps(m) == []
    T = np.atleast_2d(np.asarray(m.mat_texid))
    if T.shape[1] < 2:
        pytest.skip("this mujoco build has a scalar mat_texid; role sweep is not expressible")
    # add a cube texture in a second role on the same material
    m2 = mujoco.MjModel.from_xml_string(
        '<mujoco><asset>'
        '<texture name="a" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        '<texture name="c" type="cube" builtin="checker" rgb1="0 0 1" rgb2="1 1 0" width="8" height="48"/>'
        '<mesh name="mm" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>'
        '<material name="m" texture="a"/></asset>'
        '<worldbody><geom type="mesh" mesh="mm" group="1" material="m"/></worldbody></mujoco>')
    cube = [i for i in range(m2.ntex) if int(m2.tex_type[i]) == 1][0]
    two_d = [i for i in range(m2.ntex) if int(m2.tex_type[i]) == 0][0]
    assert texture_gaps(m2) == []
    role = np.asarray(m2.mat_texid).shape[1] - 1     # a role the 2-D texture does not occupy
    assert cube > two_d, "the cube must have the LARGER id, so np.max would have caught it anyway"
    m2.mat_texid[0, role] = two_d                    # smaller id in another role: still fine
    assert texture_gaps(m2) == []
    m2.mat_texid[0, role] = cube                     # a cube in ANY role is fatal
    assert len(texture_gaps(m2)) == 1
