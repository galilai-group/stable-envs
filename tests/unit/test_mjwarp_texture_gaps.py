"""``texture_gaps`` reports every geom whose texture MJWarp cannot sample.

MJWarp builds ``wp.Texture2D`` and samples with 2-D coordinates. A gap is (1) a 2-D texture on a
primitive geom, which has no UVs and renders one texel; (2) a non-2-D texture (cube or skybox) worn by
any geom, which is indexed out of bounds; or (3) a textured mesh without texcoords, which reads
``mesh_texcoord`` from a negative offset. ``swm/OGBScene-v0``'s cube map on UV-less button meshes hits
(2) and (3) and crashes with ``Warp CUDA error 700``. Each rule is checked on a synthetic model.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stable_envs.render.mjwarp import TEXTURE_SAMPLER_OK, TEXTURED_OK, texture_gaps

#: A tetrahedron with per-vertex UVs. Mesh fixtures carry texcoords by default, since a mesh without
#: them is a gap under condition 3.
_MESH_UV = 'vertex="0 0 0  1 0 0  0 1 0  0 0 1" texcoord="0 0  1 0  0 1  1 1"'
_MESH_NO_UV = 'vertex="0 0 0  1 0 0  0 1 0  0 0 1"'


def _model(
    tex_type: str, geom: str, *, textured: bool = True, group: int = 1, texcoord: bool = True
) -> mujoco.MjModel:
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
    size = {"plane": "1 1 .1", "mesh": "", "box": ".1 .1 .1", "sphere": ".1", "cylinder": ".1 .1", "capsule": ".1 .1"}[
        geom
    ]
    if geom == "mesh":
        verts = _MESH_UV if texcoord else _MESH_NO_UV
        asset = asset.replace("</asset>", f'<mesh name="mm" {verts}/></asset>')
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
    m = _model("2d", geom)
    assert geom != "mesh" or int(m.mesh_texcoordnum[0]) > 0, "this mesh does carry UVs"
    assert texture_gaps(m) == []


@pytest.mark.parametrize("geom", ["box", "sphere", "cylinder", "capsule"])
def test_2d_texture_on_a_primitive_is_a_gap(geom):
    """Condition 1: MJWarp cannot derive UVs, so the surface renders as one texel."""
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
    """Condition 2: a cube map is a gap even on a geom type in ``TEXTURED_OK``."""
    gaps = texture_gaps(_model("cube", geom))
    assert len(gaps) == 1, "a cube map worn by a geom must be reported whatever the geom type"
    assert gaps[0][2] == int(getattr(mujoco.mjtGeom, f"mjGEOM_{geom.upper()}"))


def test_a_cube_map_on_a_primitive_is_still_a_gap():
    assert len(texture_gaps(_model("cube", "box"))) == 1


def test_a_skybox_material_worn_by_a_geom_is_a_gap():
    """A skybox texture worn by a geom is a gap; ``texture_gaps`` only looks at ``geom_matid``."""
    assert len(texture_gaps(_model("skybox", "mesh"))) == 1


def test_the_models_own_skybox_is_not_flagged():
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><asset>"
        '<texture name="sky" type="skybox" builtin="gradient" width="8" height="48"/>'
        '<texture name="t" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        '<material name="m" texture="t"/></asset>'
        '<worldbody><geom type="plane" size="1 1 .1" group="1" material="m"/></worldbody></mujoco>'
    )
    assert int(m.ntex) == 2 and any(int(m.tex_type[i]) == 2 for i in range(m.ntex))
    assert texture_gaps(m) == [], "a skybox no geom wears is not a geom's texture gap"


def test_every_texture_role_is_examined_not_only_the_largest_id():
    """Every ``mat_texid`` role is examined: a cube map in any role is a gap, a 2-D texture is not."""
    m = _model("2d", "mesh")
    assert texture_gaps(m) == []
    T = np.atleast_2d(np.asarray(m.mat_texid))
    if T.shape[1] < 2:
        pytest.skip("this mujoco build has a scalar mat_texid; role sweep is not expressible")
    # Add a cube texture in a second role on the same material.
    m2 = mujoco.MjModel.from_xml_string(
        "<mujoco><asset>"
        '<texture name="a" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        '<texture name="c" type="cube" builtin="checker" rgb1="0 0 1" rgb2="1 1 0" width="8" height="48"/>'
        f'<mesh name="mm" {_MESH_UV}/>'
        '<material name="m" texture="a"/></asset>'
        '<worldbody><geom type="mesh" mesh="mm" group="1" material="m"/></worldbody></mujoco>'
    )
    cube = [i for i in range(m2.ntex) if int(m2.tex_type[i]) == 1][0]
    two_d = [i for i in range(m2.ntex) if int(m2.tex_type[i]) == 0][0]
    assert texture_gaps(m2) == []
    role = np.asarray(m2.mat_texid).shape[1] - 1  # a role the 2-D texture does not occupy
    assert cube > two_d, "the fixture expects the cube texture to have the larger id"
    m2.mat_texid[0, role] = two_d  # smaller id in another role: still fine
    assert texture_gaps(m2) == []
    m2.mat_texid[0, role] = cube  # a cube in any role is a gap
    assert len(texture_gaps(m2)) == 1


def test_a_textured_mesh_with_no_texcoords_is_a_gap():
    """Condition 3, checked with a 2-D texture so it is independent of condition 2.

    ``sample_texture``'s mesh branch indexes ``mesh_texcoord`` without a guard, so a textured mesh with
    no texcoords reads from a negative offset.
    """
    m = _model("2d", "mesh", texcoord=False)
    assert int(m.mesh_texcoordnum[0]) == 0, "this fixture deliberately carries no UVs"
    assert len(texture_gaps(m)) == 1, "a textured mesh with no texcoords must be reported"


def test_an_untextured_mesh_without_texcoords_is_not_a_gap():
    """No texture, no sampling, no crash — the condition must not fire on geometry alone."""
    m = _model("2d", "mesh", textured=False, texcoord=False)
    assert int(m.mesh_texcoordnum[0]) == 0
    assert texture_gaps(m) == []
