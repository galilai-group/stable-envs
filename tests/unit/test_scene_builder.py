import pytest
import torch

from stable_envs.render.scene import SceneBuilder


@pytest.mark.parametrize("plural", [False, True])
def test_polygons_reject_excess_vertices(plural):
    builder = SceneBuilder(2, vertices=4)
    with pytest.raises(ValueError, match="vertices"):
        if plural:
            builder.add_polys(torch.zeros(2, 3, 5, 2), [255, 0, 0])
        else:
            builder.add_poly(torch.zeros(5, 2), [255, 0, 0])


def test_explicit_primitive_colors_when_primitive_count_equals_batch():
    builder = SceneBuilder(2)
    colors = torch.tensor([[[255, 0, 0], [0, 255, 0]]])
    builder.add_rects(torch.zeros(2, 2, 2), torch.ones(2, 2, 2), colors)
    scene = builder.build([0, 0, 0], [-1, -1, 1, 1])
    assert scene.fill.device.type == "cpu"
    assert torch.equal(scene.fill, colors.expand(2, 2, 3))
    scene.fill = scene.fill[:, :1]
    with pytest.raises(ValueError, match="fill"):
        scene.validate()


def test_invalid_scene_and_geometry_fail_explicitly():
    with pytest.raises(ValueError, match="no primitives"):
        SceneBuilder(1).build(0, [0, 0, 1, 1])
    builder = SceneBuilder(1, vertices=2)
    with pytest.raises(ValueError, match="capacity"):
        builder.add_segment([0, 0], [1, 1], 0.1, [255, 0, 0])
    builder = SceneBuilder(1)
    builder.add_rect([0, 0], [1, 1], [255, 0, 0])
    with pytest.raises(ValueError, match="extent"):
        builder.build(0, [0, 0, 0, 1])
