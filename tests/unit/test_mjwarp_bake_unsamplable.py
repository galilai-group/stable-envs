"""``bake_unsamplable_textures`` flattens textures MJWarp cannot sample to their mean colour.

It covers textures that crash the sampler (e.g. ``swm/OGBScene-v0``'s cube map on UV-less button meshes,
which raises ``Warp CUDA error 700``), and leaves merely badly UV'd geoms, which still render, alone.
The bake returns a record that names what it removed.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stable_envs.render.mjwarp import (
    bake_unsamplable_textures,
    restore_primitive_textures,
    texture_gaps,
    unsamplable_texture_materials,
)

_UV = 'vertex="0 0 0  1 0 0  0 1 0  0 0 1" texcoord="0 0  1 0  0 1  1 1"'
_NO_UV = 'vertex="0 0 0  1 0 0  0 1 0  0 0 1"'
_T2D = '<texture name="t" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
_TCUBE = '<texture name="t" type="cube" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="48"/>'


def _model(body: str, asset: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><asset>{asset}<material name="m" texture="t"/></asset><worldbody>{body}</worldbody></mujoco>'
    )


def _cube_on_uvless_mesh():
    return _model('<geom type="mesh" mesh="mm" group="1" material="m"/>', _TCUBE + f'<mesh name="mm" {_NO_UV}/>')


def _2d_on_uvless_mesh():
    return _model('<geom type="mesh" mesh="mm" group="1" material="m"/>', _T2D + f'<mesh name="mm" {_NO_UV}/>')


def _2d_on_uv_mesh():
    return _model('<geom type="mesh" mesh="mm" group="1" material="m"/>', _T2D + f'<mesh name="mm" {_UV}/>')


def _2d_on_box():
    return _model('<geom type="box" size=".1 .1 .1" group="1" material="m"/>', _T2D)


def test_a_cube_map_on_a_uvless_mesh_is_baked():
    m = _cube_on_uvless_mesh()
    assert unsamplable_texture_materials(m) == {0: [0]}
    b = bake_unsamplable_textures(m)
    assert b["n_materials"] == 1 and b["n_geoms"] == 1
    assert int(m.mat_texid[0][1]) == -1 if np.ndim(m.mat_texid[0]) else int(m.mat_texid[0]) == -1
    assert texture_gaps(m) == [], "after the bake the scene is samplable"


def test_a_2d_texture_on_a_uvless_mesh_is_baked_too():
    """Any texture on a UV-less mesh is unsamplable, not only a cube map."""
    m = _2d_on_uvless_mesh()
    assert unsamplable_texture_materials(m) == {0: [0]}
    assert bake_unsamplable_textures(m)["n_geoms"] == 1


def test_a_textured_mesh_with_uvs_is_left_alone():
    m = _2d_on_uv_mesh()
    assert unsamplable_texture_materials(m) == {}
    before = np.array(m.mat_rgba, copy=True), np.array(m.mat_texid, copy=True)
    b = bake_unsamplable_textures(m)
    assert b["n_materials"] == 0
    assert np.array_equal(m.mat_rgba, before[0]) and np.array_equal(m.mat_texid, before[1])


def test_a_badly_uvd_primitive_is_NOT_this_helpers_business():
    """A textured box is a UV gap but still renders, so this helper leaves it alone."""
    m = _2d_on_box()
    assert texture_gaps(m), "the box IS a UV gap"
    assert unsamplable_texture_materials(m) == {}, "but it is not an unsamplable one"
    assert bake_unsamplable_textures(m)["n_materials"] == 0


def test_groups_are_respected():
    m = _cube_on_uvless_mesh()
    assert bake_unsamplable_textures(m, groups=(0,))["n_materials"] == 0


def test_the_bake_is_reversible():
    m = _cube_on_uvless_mesh()
    rgba, texid = np.array(m.mat_rgba, copy=True), np.array(m.mat_texid, copy=True)
    b = bake_unsamplable_textures(m)
    assert not np.array_equal(m.mat_texid, texid)
    restore_primitive_textures(m, b)
    assert np.array_equal(m.mat_rgba, rgba) and np.array_equal(m.mat_texid, texid)


def test_the_record_is_declarable_not_just_a_count():
    """The bake record names the material, texture, type, mean colour and geoms."""
    m = _cube_on_uvless_mesh()
    rec = bake_unsamplable_textures(m)["baked"][0]
    for k in ("material", "material_name", "texture", "texture_name", "tex_type", "mean_rgb", "geoms", "geom_names"):
        assert k in rec, f"a declaration needs {k}"
    assert rec["tex_type"] == int(mujoco.mjtTexture.mjTEXTURE_CUBE)
    assert len(rec["mean_rgb"]) == 3 and all(0.0 <= c <= 1.0 for c in rec["mean_rgb"])


def test_the_mean_colour_is_actually_the_textures_mean():
    """A red/green checker averages to something between the two."""
    m = _cube_on_uvless_mesh()
    r, g, b_ = bake_unsamplable_textures(m)["baked"][0]["mean_rgb"]
    assert 0.2 < r < 0.8 and 0.2 < g < 0.8, (r, g, b_)


def test_an_untextured_model_is_a_no_op():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1" group="1"/></worldbody></mujoco>'
    )
    assert unsamplable_texture_materials(m) == {}
    assert bake_unsamplable_textures(m)["n_materials"] == 0


def test_the_record_names_EVERY_removed_texture_not_just_the_rgb_role():
    """The record names every texture role the bake clears, not only the RGB role.

    A material can wear a 2-D albedo in the RGB role and a cube map in another; the bake clears both.
    """
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><asset>"
        '<texture name="a" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="8" height="8"/>'
        '<texture name="c" type="cube" builtin="checker" rgb1="0 0 1" rgb2="1 1 0" width="8" height="48"/>'
        f'<mesh name="mm" {_NO_UV}/>'
        '<material name="m" texture="a"/></asset>'
        '<worldbody><geom type="mesh" mesh="mm" group="1" material="m"/></worldbody></mujoco>'
    )
    T = np.atleast_2d(np.asarray(m.mat_texid))
    if T.shape[1] < 2:
        pytest.skip("scalar mat_texid on this mujoco build; multi-role is not expressible")
    cube = [i for i in range(m.ntex) if int(m.tex_type[i]) == 1][0]
    m.mat_texid[0, T.shape[1] - 1] = cube  # a cube map in a second role
    rec = bake_unsamplable_textures(m)["baked"][0]
    assert rec["n_removed"] == 2, "both roles were cleared, so both must be recorded"
    types = sorted(r["tex_type"] for r in rec["removed_textures"])
    assert types == [0, 1], f"the record must name the 2-D and the cube texture, got {types}"
    assert any(r["unsamplable"] for r in rec["removed_textures"]), (
        "the record must say which removed texture the sampler could not address"
    )


def test_the_single_role_record_still_names_its_texture():
    m = _cube_on_uvless_mesh()
    rec = bake_unsamplable_textures(m)["baked"][0]
    assert rec["n_removed"] >= 1
    assert rec["removed_textures"][0]["tex_type"] == int(mujoco.mjtTexture.mjTEXTURE_CUBE)
