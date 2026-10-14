"""Batch bookkeeping checks without a graphics context; GPU parity is gated separately."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.helpers.ogbench_opengl import MODEL_VALUES, opengl, state  # noqa: F401


@pytest.mark.parametrize("family", ["Cube", "Scene"])
@pytest.mark.parametrize("supersample", [1, 2])
def test_grouped_render_keeps_order_state_and_one_forward(opengl, family, supersample):  # noqa: F811
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


def test_grouping_includes_every_compiled_value_and_exact_float_bytes(opengl):  # noqa: F811
    module, _ = opengl
    rs = state()
    assert list(module._configuration_groups(rs)) == [[0, 2], [1, 4], [3]]
    for name in MODEL_VALUES:
        changed = rs.clone()
        changed["appearance", name][2, 0] = -0.0 if name != "cube.size" else 7
        want = [[0, 2], [1, 4], [3]] if name == "agent.color" else [[0], [1, 4], [2], [3]]
        assert list(module._configuration_groups(changed)) == want
