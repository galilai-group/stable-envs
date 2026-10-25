"""Alpha events preserve overlap, exact ties and opaque termination."""

import numpy as np
import pytest

from scripts.testing.object_permanence_alpha_reference import draw
from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera
from tests.unit.test_object_permanence_ray_contract import camera, quad


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("alpha", [0, 0.25, 1])
def test_source_alpha_events_count_shared_edges_once_and_stop_at_opaque(alpha, orthographic):
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    meshes = {"a_glass": quad(-2)[faces], "b_ball": quad(-3)[faces], "c_background": quad(-4)[faces]}
    r = draw(
        meshes,
        dict(a_glass=alpha, b_ball=1, c_background=1),
        orthographic_camera() if orthographic else camera(),
        31,
        47,
        max_layers=4,
    )
    expected = [2, 0, 0, 0] if alpha == 0 else [1, 2, 0, 0] if alpha < 1 else [1, 0, 0, 0]
    np.testing.assert_array_equal(r["layer_ids"][15, 23], expected)
    expected_depth = [3, 0, 0, 0] if alpha == 0 else [2, 3, 0, 0] if alpha < 1 else [2, 0, 0, 0]
    np.testing.assert_array_equal(r["layer_depth"][15, 23], expected_depth)


def test_source_coplanar_distinct_objects_keep_both_events_and_capacity_is_not_a_cutoff():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    meshes = {"a": quad(-2)[faces], "b": quad(-2)[faces], "c": quad(-3)[faces]}
    r = draw(meshes, dict(a=0.3, b=0.4, c=1), camera(), 31, 47, max_layers=3)
    np.testing.assert_array_equal(r["layer_ids"][15, 23], [1, 2, 3])
    with pytest.raises(ValueError, match="capacity"):
        draw(meshes, dict(a=0.3, b=0.4, c=1), camera(), 31, 47, max_layers=2)
