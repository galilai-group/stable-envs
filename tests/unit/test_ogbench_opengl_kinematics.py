"""Guard and routing regressions for Scene's rendering-only pose update."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_ogbench_opengl_batch import opengl as opengl
from test_ogbench_opengl_batch import state


def context(mujoco):
    """A fixed scene with real NumPy model/color buffers and default options."""
    mujoco.mjtEnableBit = SimpleNamespace(mjENBL_SLEEP=16)
    mujoco.mjtCamLight = SimpleNamespace(mjCAMLIGHT_FIXED=0)
    mujoco.mjtWrap = SimpleNamespace(mjWRAP_JOINT=1)
    mujoco.MjvOption = lambda: SimpleNamespace(flags=np.zeros(32, dtype=bool), frame=0, label=0)
    mujoco.mj_kinematics = Mock()
    mujoco.mj_camlight = Mock()
    materials = {
        name: SimpleNamespace(rgba=np.zeros(4))
        for name in ["drawer_handle", "window_handle", "ur5e/robotiq/black", "ur5e/robotiq/pad_gray"]
    }
    model = SimpleNamespace(
        nflex=0,
        nskin=0,
        nplugin=0,
        nsensor=0,
        opt=SimpleNamespace(enableflags=0),
        cam_mode=np.zeros(2, dtype=int),
        light_mode=np.zeros(2, dtype=int),
        wrap_type=np.ones(2, dtype=int),
        geom_rgba=np.zeros((4, 4)),
        material=materials.__getitem__,
    )
    env = SimpleNamespace(
        model=model,
        data=SimpleNamespace(qpos=np.zeros(1), mocap_pos=np.zeros(3), mocap_quat=np.zeros(4)),
        _cur_button_states=np.zeros(2, dtype=int),
        _colors={"white": np.ones(4)},
        _button_geom_ids_list=[[1, 2], [3]],
        _cube_target_geom_ids_list=[[0]],
        _cube_colors_from_variation=Mock(),
        variation_space=SimpleNamespace(_walk=lambda name: SimpleNamespace(_value=None)),
    )
    env._apply_button_states = Mock(side_effect=lambda: mujoco.mj_forward(model, env.data))
    renderer = SimpleNamespace(_scene_option=mujoco.MjvOption(), update_scene=Mock())
    return env, renderer


@pytest.mark.parametrize(
    "field",
    [
        "nflex",
        "nskin",
        "nplugin",
        "nsensor",
        "sleep",
        "cam_mode",
        "light_mode",
        "wrap_type",
        "flags",
        "frame",
        "label",
    ],
)
def test_unsupported_features_require_full_forward(opengl, field):  # noqa: F811
    module, mujoco = opengl
    env, renderer = context(mujoco)
    assert module._can_use_render_kinematics(env, renderer)
    if field in ["nflex", "nskin", "nplugin", "nsensor"]:
        setattr(env.model, field, 1)
    elif field == "sleep":
        env.model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_SLEEP
    elif field in ["cam_mode", "light_mode", "wrap_type"]:
        getattr(env.model, field)[0] = 3
    elif field == "flags":
        renderer._scene_option.flags[0] = True
    else:
        setattr(renderer._scene_option, field, 1)
    assert not module._can_use_render_kinematics(env, renderer)


@pytest.mark.parametrize("buttons", [[0, 0], [0, 1], [1, 0], [1, 1]])
def test_button_colors_include_every_geom_and_handle(opengl, buttons):  # noqa: F811
    module, mujoco = opengl
    env, _ = context(mujoco)
    env._cur_button_states[:] = buttons
    colors = np.array([[0.2, 0.3, 0.4], [0.5, 0.5, 0.5]])
    saved = colors.copy()
    env.model.geom_rgba[0] = [0.1, 0.2, 0.3, 0.2]
    unrelated_cube = env.model.geom_rgba[0].copy()
    module._apply_render_button_colors(env, colors)
    for index, name in enumerate(["drawer_handle", "window_handle"]):
        rgba = np.r_[colors[index] if buttons[index] == 0 else np.ones(3), 1]
        for geom in env._button_geom_ids_list[index]:
            np.testing.assert_array_equal(env.model.geom_rgba[geom], rgba)
        np.testing.assert_array_equal(env.model.material(name).rgba, rgba)
    np.testing.assert_array_equal(colors, saved)
    np.testing.assert_array_equal(env.model.geom_rgba[0], unrelated_cube)
    mujoco.mj_forward.assert_not_called()


@pytest.mark.parametrize(
    "family,unsupported,fast", [("Scene", False, True), ("Scene", True, False), ("Cube", False, False)]
)
def test_pose_updates_use_fast_path_only_for_supported_scene(opengl, family, unsupported, fast):  # noqa: F811
    module, mujoco = opengl
    env, renderer = context(mujoco)
    if unsupported:
        renderer._scene_option.flags[0] = True
    events = []
    mujoco.mj_kinematics.side_effect = lambda *args: events.append("kinematics")
    mujoco.mj_camlight.side_effect = lambda *args: events.append("camlight")
    renderer.update_scene.side_effect = lambda *args, **kwargs: events.append("scene")

    def draw(*, out):
        events.append("draw")
        out[:] = int(env.data.qpos[0])
        return out

    renderer.render = draw
    module._renderer = Mock(return_value=(env, renderer, "Mesa", "llvmpipe"))
    rs = state()
    saved = rs.clone()
    image = module.render(family, rs, "cpu")
    assert torch.equal(image[:, 0, 0, 0], torch.arange(5, dtype=torch.uint8))
    assert all(torch.equal(rs[key], value) for key, value in saved.items(True, True))
    assert mujoco.mj_kinematics.call_count == (5 if fast else 0)
    assert mujoco.mj_camlight.call_count == (5 if fast else 0)
    assert mujoco.mj_forward.call_count == (0 if fast else 5)
    assert env._apply_button_states.call_count == (5 if family == "Scene" and not fast else 0)
    assert events == (["kinematics", "camlight", "scene", "draw"] if fast else ["scene", "draw"]) * 5
