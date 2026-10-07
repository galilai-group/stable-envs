import numpy as np
import pytest
import torch

from stable_envs.render import cpu3d

mujoco = pytest.importorskip("mujoco")


def reflective_scene(tilt=0, reflectance=0.4, shadows=False):
    light = '<light pos="-1 -1 4" dir="0 0 -1" cutoff="80"/>' if shadows else ""
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
    <asset>
      <material name="floor" rgba="0.2 0.4 0.6 1" emission="1" reflectance="{reflectance}"/>
      <material name="ball" rgba="0.8 0.2 0.1 1" emission="1"/>
    </asset><worldbody>{light}
      <camera pos="0 -4 3" xyaxes="1 0 0 0 0.6 0.8"/>
      <geom type="plane" size="4 4 0.1" euler="{tilt} 0 0" material="floor"/>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size="0.3" material="ball"/></body>
    </worldbody></mujoco>""")
    scene = cpu3d.Scene(model, 0, 64, [1], use_shadows=shadows)
    data = mujoco.MjData(model)
    rows = []
    for x in [-0.4, 0, 0.5]:
        data.qpos[0] = x
        mujoco.mj_forward(model, data)
        rows.append(
            (
                data.cam_xpos.copy()[0],
                data.geom_xpos[1:].copy(),
                data.geom_xmat[1:].reshape(1, 3, 3).copy(),
                data.light_xpos.copy(),
                data.light_xdir.copy(),
            )
        )
    args = [torch.tensor(np.stack([row[i] for row in rows]), dtype=torch.float32) for i in range(5)]
    materials = torch.tensor(model.mat_rgba[:, :3].copy()).expand(3, -1, -1).clone()
    materials[:, 1, 0] = torch.tensor([0.6, 0.7, 0.8])
    lights = torch.tensor(model.light_diffuse.copy()).expand(3, -1, -1).clone()
    return scene, (*args, materials, lights)


@pytest.mark.parametrize("tilt", [0, 25])
def test_plane_reflections_match_analytic_sphere_and_material_addition(tilt):
    scene, args = reflective_scene(tilt)
    result = cpu3d.render(scene, *args).numpy().reshape(3, -1, 3)
    scene.plane_reflectance = 0
    baseline = cpu3d.render(scene, *args).numpy().reshape(3, -1, 3)
    origin, centers = args[0].numpy(), args[1].numpy()[:, 0]
    rays = scene.dirs.astype(np.float64)
    normal = scene.plane_mat[:, 2]
    normal_dot = (rays * normal).sum(-1)
    distance = ((scene.plane_pos - origin) * normal).sum(-1)[:, None] / normal_dot
    points = origin[:, None] + distance[..., None] * rays
    reflected = rays - 2 * normal_dot[:, None] * normal

    def sphere_distance(start, direction):
        offset = start - centers[:, None]
        b = (offset * direction).sum(-1)
        discriminant = b * b - ((offset * offset).sum(-1) - 0.3**2)
        near = -b - np.sqrt(discriminant.clip(0))
        return np.where((discriminant >= 0) & (near >= 0), near, np.inf)

    primary = sphere_distance(origin[:, None], rays)
    mirror = sphere_distance(points, reflected)
    local = np.einsum("bni,ij->bnj", points - scene.plane_pos, scene.plane_mat)
    visible = (distance >= 0) & (normal_dot < 0) & (distance < primary)
    visible &= (abs(local[..., 0]) <= 4) & (abs(local[..., 1]) <= 4)
    expected_hits = visible & np.isfinite(mirror)
    assert expected_hits.sum() > 20
    assert np.array_equal((result != baseline).any(-1), expected_hits)
    for lane in range(3):
        expected = (0.4 * args[5][lane, 1] + args[5][lane, 0]).clamp(0, 1).numpy() * 255
        assert np.max(abs(result[lane, expected_hits[lane]].astype(float) - expected)) <= 1


@pytest.mark.parametrize("shadows", [False, True])
def test_reflected_bounds_chunks_and_output_ownership(monkeypatch, shadows):
    scene, args = reflective_scene(shadows=shadows)
    expected = cpu3d.render(scene, *args)
    assert torch.equal(expected, cpu3d.render(scene, *args, chunk_size=1))
    monkeypatch.setattr(cpu3d, "FORCE_FULL_FRAME", True)
    assert torch.equal(expected, cpu3d.render(scene, *args))
    saved = expected.clone()
    args[5][:, 1] = 0
    assert not torch.equal(expected, cpu3d.render(scene, *args))
    assert torch.equal(expected, saved)


@pytest.mark.parametrize("tilt", [0, 25])
def test_plane_footprint_matches_numeric_projection_derivatives(tilt):
    scene, args = reflective_scene(tilt)
    scene.plane_texscale = np.array([3.0, 5.0])
    gradients = scene._plane_uv_gradients().numpy()
    camera = args[0][0].numpy().astype(float)
    rotation = scene.xmat
    normal = scene.plane_mat[:, 2]
    height = ((scene.plane_pos - camera) * normal).sum()
    pixel = 2 * np.tan(np.deg2rad(scene.fovy) / 2) / scene.res

    def uv(ray):
        distance = height / (ray * normal).sum()
        point = camera + distance * ray - scene.plane_pos
        return np.array([(point * scene.plane_mat[:, 0]).sum() * 1.5, -(point * scene.plane_mat[:, 1]).sum() * 2.5])

    for index in [500, 2000, 4095]:
        ray = scene.dirs[index].astype(float)
        ray /= -(ray * rotation[:, 2]).sum()
        expected = []
        for delta in (pixel * rotation[:, 0], -pixel * rotation[:, 1]):
            expected.extend((uv(ray + delta * 1e-3) - uv(ray - delta * 1e-3)) / 2e-3)
        np.testing.assert_allclose(gradients[index] * height, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "size,repeat,uniform,version,scale,offset",
    [
        ((250, 0.8), (1, 1), True, (3, 11), (1, 1), (-0.5, -0.5)),
        ((250, 0.8), (1, 1), True, (3, 12), (1, 1), (125, 0.4)),
        ((0, 0), (2, 3), True, (3, 11), (2, 3), (-0.5, -0.5)),
        ((0, 0), (2, 3), True, (3, 12), (2, 3), (-0.5, -0.5)),
        ((4, 0), (2, 3), True, (3, 12), (2, 3), (3.5, -0.5)),
        ((4, 2), (2, 3), False, (3, 11), (0.5, 1.5), (-0.5, -0.5)),
        ((4, 2), (2, 3), False, (3, 12), (0.5, 1.5), (1, 1.5)),
        ((4, 2), (0, -2), True, (3, 12), (1, 1), (2, 1)),
    ],
)
def test_plane_texgen_respects_source_version_and_plane_extent(size, repeat, uniform, version, scale, offset):
    actual_scale, actual_offset = cpu3d.plane_texgen(size, repeat, uniform, version)
    np.testing.assert_allclose(actual_scale, scale)
    np.testing.assert_allclose(actual_offset, offset)


def test_blocked_direct_light_preserves_headlight_ambient():
    scene, rows = reflective_scene(shadows=True)
    camera, position, rotation, light_pos, light_dir, materials, diffuse = [x[1:2].clone() for x in rows]
    light_pos[:] = torch.tensor([0.0, 0.0, 4.0])
    scene.headlight = (np.array([0.1, 0.2, 0.3]), np.zeros(3), np.zeros(3))
    scene.mat_emis.zero_()
    scene.mat_spec.zero_()
    base = materials[:, 0]
    result = cpu3d._shade_hits(
        scene,
        base,
        -camera,
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, -0.8, 0.6]]),
        torch.tensor([0]),
        torch.tensor([0]),
        camera,
        position - camera[:, None],
        rotation,
        light_pos,
        light_dir,
        diffuse,
    )
    torch.testing.assert_close(result, base * torch.tensor([[0.1, 0.2, 0.3]]), rtol=0, atol=1e-7)


def test_floor_light_interpolates_source_triangle_vertices():
    scene, rows = reflective_scene(shadows=True)
    camera, position, rotation, light_pos, light_dir, materials, diffuse = [x[1:2].clone() for x in rows]
    scene.plane_grid = (torch.tensor([-4.0, 0.0, 4.0]), torch.tensor([-4.0, 0.0, 4.0]))
    scene.headlight_active = False
    scene.lights[0].exponent = 0
    scene.mat_emis.zero_()
    scene.mat_spec.zero_()
    light_pos[:] = torch.tensor([0.0, 0.0, 4.0])
    point = torch.tensor([[-3.0, 2.0, 0.0]]) - camera
    base = materials[:, 0]
    result = cpu3d._shade_hits(
        scene,
        base,
        point,
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, -0.8, 0.6]]),
        torch.tensor([0]),
        torch.tensor([0]),
        camera,
        position - camera[:, None],
        rotation,
        light_pos,
        light_dir,
        diffuse,
        shadows=False,
    )
    # The containing triangle has weights .5/.25/.25 at (-4,0), (0,4), (-4,4).
    vertex_lighting = 0.75 * 4 / 32**0.5 + 0.25 * 4 / 48**0.5
    expected = base * diffuse[:, 0] * vertex_lighting
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-7)
    per_pixel = base * diffuse[:, 0] * (4 / 29**0.5)
    assert torch.max(abs(result - per_pixel)) > 0.01
