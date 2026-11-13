import numpy as np
import pytest
import torch

from stable_envs.render import cpu3d

mujoco = pytest.importorskip("mujoco")


def scene_inputs(*, tilt=0, shadows=True, specular=True, size=42):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><asset>
      <texture name="floor" type="2d" builtin="checker" width="16" height="16"
               rgb1="0.2 0.3 0.4" rgb2="0.7 0.8 0.9"/>
      <texture type="skybox" builtin="gradient" width="16" height="96"
               rgb1="0.1 0.2 0.3" rgb2="0.8 0.9 1"/>
      <material name="floor" texture="floor" specular="0.3"/>
      <material name="body" rgba="0.7 0.2 0.1 1" specular="0.8" shininess="0.4"/>
    </asset><worldbody>
      <camera pos="0 -4 3" xyaxes="1 0 0 0 0.6 0.8"/>
      <light pos="-1 -1 4" dir="0 0 -1" cutoff="80"/>
      <geom type="plane" size="4 4 0.1" euler="{tilt} 0 0" material="floor"/>
      <body pos="0 0 0.8"><freejoint/>
        <geom type="sphere" size="0.25" material="body"/>
        <geom type="capsule" size="0.12 0.3" pos="0.65 0 0" material="body"/>
        <geom type="box" size="0.2 0.15 0.3" pos="-0.65 0 0" rgba="0.1 0.7 0.3 1"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    scene = cpu3d.Scene(model, 0, size, [1, 2, 3], use_shadows=shadows, enable_specular=specular)
    poses = []
    for i in range(19):
        data.qpos[0] = (i - 9) * 0.03
        data.qpos[2] = 0.8 + (i % 3) * 0.15
        mujoco.mj_forward(model, data)
        poses.append(
            (
                data.cam_xpos[0].copy(),
                data.geom_xpos[1:].copy(),
                data.geom_xmat[1:].reshape(3, 3, 3).copy(),
                data.light_xpos.copy(),
                data.light_xdir.copy(),
            )
        )
    args = [torch.tensor(np.stack([p[k] for p in poses]), dtype=torch.float32) for k in range(5)]
    material = torch.tensor(model.mat_rgba[:, :3].copy()).expand(19, -1, -1).clone()
    material[:, 1, 0] = torch.linspace(0.2, 0.8, 19)
    lights = torch.tensor(model.light_diffuse.copy()).expand(19, -1, -1).clone()
    return scene, (*args, material, lights)


@pytest.mark.parametrize("specular", [False, True])
def test_cpu_scene_bounds_and_chunks_preserve_pixels(monkeypatch, specular):
    scene, args = scene_inputs(specular=specular)
    expected = cpu3d.render(scene, *args, backend="torch")
    assert expected.shape == (19, 42, 42, 3)
    assert torch.unique(expected.reshape(-1, 3), dim=0).shape[0] > 100
    assert torch.equal(expected, cpu3d.render(scene, *args, chunk_size=1, backend="torch"))
    monkeypatch.setattr(cpu3d, "FORCE_FULL_FRAME", True)
    monkeypatch.setattr(cpu3d, "SHADOW_TEST_ALL_HITS", True)
    assert torch.equal(expected, cpu3d.render(scene, *args, backend="torch"))


@pytest.mark.parametrize("tilt", [0.1, 30, 180])
def test_tilted_shadow_floor_is_refused(tilt):
    with pytest.raises(ValueError, match="horizontal.*plane"):
        scene_inputs(tilt=tilt)


def test_tilted_floor_without_shadows_still_renders():
    scene, args = scene_inputs(tilt=30, shadows=False)
    frame = cpu3d.render(scene, *args)
    assert frame.shape == (19, 42, 42, 3)
    assert frame.dtype == torch.uint8
