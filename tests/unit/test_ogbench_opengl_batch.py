"""Batch bookkeeping checks without a graphics context; GPU parity is gated separately."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

MODEL_VALUES = ("cube.size", "agent.color", "floor.color", "camera.angle_delta", "light.intensity")


@pytest.fixture
def opengl():
    """Load the real renderer with optional environment packages replaced at import."""
    pixels = ModuleType("stable_envs.envs.ogbench.simulation.render_state")
    pixels.validate = Mock()
    pixels.AGENT_MATERIALS = ("ur5e/robotiq/black", "ur5e/robotiq/pad_gray")
    compat = ModuleType("stable_envs.envs.ogbench.simulation.compatibility")
    compat.MODEL_VALUES = MODEL_VALUES
    compat.CompatibleCube = compat.CompatibleScene = object
    resources = ModuleType("stable_envs.render._resources")
    resources.retain_process_lifetime = Mock()
    mujoco = ModuleType("mujoco")
    mujoco.mj_forward = Mock()
    path = Path(__file__).parents[2] / "stable_envs/envs/ogbench/manipulation_opengl.py"
    path = Path(os.environ.get("STABLE_ENVS_OPENGL_TEST_CANDIDATE", path))
    spec = importlib.util.spec_from_file_location("stable_envs.envs.ogbench._opengl_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {pixels.__name__: pixels, compat.__name__: compat, resources.__name__: resources, "mujoco": mujoco},
    ):
        spec.loader.exec_module(module)
        # A prior test may have populated the parent package attribute already.
        module.pixels = pixels
        module.version = Mock(side_effect=lambda name: {"mujoco": "3.12.0", "ogbench": "1.2.1"}[name])
        yield module, mujoco


def state(supersample=1):
    b = 5
    values = {
        ("renderer", "size"): torch.full((b, 1), 2),
        ("renderer", "supersample"): torch.full((b, 1), supersample),
        ("pose", "qpos"): torch.arange(b, dtype=torch.float64).reshape(b, 1),
        ("pose", "mocap_pos"): torch.zeros(b, 3, dtype=torch.float64),
        ("pose", "mocap_quat"): torch.zeros(b, 4, dtype=torch.float64),
        ("pose", "buttons"): torch.zeros(b, 2, dtype=torch.int64),
        ("appearance", "cube.color"): torch.zeros(b, 1, 3, dtype=torch.float64),
        ("appearance", "lock_color"): torch.zeros(b, 2, 3, dtype=torch.float64),
    }
    for name in MODEL_VALUES:
        values["appearance", name] = torch.zeros(b, 1, dtype=torch.float64)
    values["appearance", "cube.size"][:, 0] = torch.tensor([1, 2, 1, 3, 2])
    return TensorDict(values, [b])


@pytest.mark.parametrize("family", ["Cube", "Scene"])
@pytest.mark.parametrize("supersample", [1, 2])
def test_grouped_render_keeps_order_state_and_one_forward(opengl, family, supersample):
    module, mujoco = opengl
    rs = state(supersample)
    saved = rs.clone()
    visits = []
    # This exercises the full-forward fallback; fast Scene routing has its own tests.
    module._can_use_render_kinematics = Mock(return_value=False)

    def renderer(family, row, resolution):
        env = SimpleNamespace(
            data=SimpleNamespace(qpos=np.zeros(1), mocap_pos=np.zeros(3), mocap_quat=np.zeros(4)),
            model=SimpleNamespace(geom_rgba=np.zeros((1, 4)), material=lambda name: SimpleNamespace(rgba=np.zeros(4))),
            variation_space=SimpleNamespace(_walk=lambda name: SimpleNamespace(_value=None)),
            _cube_colors_from_variation=lambda: None,
            _cube_target_geom_ids_list=[[0]],
        )
        env._apply_button_states = lambda: mujoco.mj_forward(env.model, env.data)

        def draw(*, out):
            visits.append(int(env.data.qpos[0]))
            out[:] = int(env.data.qpos[0])
            return out

        return env, SimpleNamespace(render=draw, update_scene=Mock()), "Mesa", "llvmpipe"

    module._renderer = Mock(side_effect=renderer)
    out = module.render(family, rs, "cpu")
    assert visits == [0, 2, 1, 4, 3]
    assert module._renderer.call_count == 3
    assert mujoco.mj_forward.call_count == 5
    assert torch.equal(out[:, 0, 0, 0], torch.arange(5, dtype=torch.uint8))
    assert out.shape == (5, 2, 2, 3)
    assert all(torch.equal(rs[key], value) for key, value in saved.items(True, True))
    previous = out.clone()
    rs["pose", "qpos"].add_(10)
    again = module.render(family, rs, "cpu")
    assert torch.equal(out, previous), "A later render must not overwrite returned image storage"
    assert torch.equal(again[:, 0, 0, 0], torch.arange(10, 15, dtype=torch.uint8))
    assert module.version.call_count == 2


def test_grouping_includes_every_compiled_value_and_exact_float_bytes(opengl):
    module, _ = opengl
    rs = state()
    assert list(module._configuration_groups(rs)) == [[0, 2], [1, 4], [3]]
    for name in MODEL_VALUES:
        changed = rs.clone()
        changed["appearance", name][2, 0] = -0.0 if name != "cube.size" else 7
        want = [[0, 2], [1, 4], [3]] if name == "agent.color" else [[0], [1, 4], [2], [3]]
        assert list(module._configuration_groups(changed)) == want
