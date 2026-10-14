"""The OGBench OpenGL renderer loaded with its optional packages stubbed, and a small render State."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

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
