"""The recovered public Cube/Scene configuration must agree with its readiness record."""

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import execution_for, status_for


@pytest.mark.parametrize("family,count,status", [("Cube", 11, "READY"), ("Scene", 8, "PARTIAL")])
def test_live_recovered_configuration_matches_readiness(family, count, status):
    pytest.importorskip("ogbench")
    from stable_envs.envs.ogbench import OGBCube, OGBScene

    cls = OGBCube if family == "Cube" else OGBScene
    with cls(seed=17) as env:
        mapping = VARIATION_MAPS[cls.__name__]
        assert set(env.variation_space.names()) == set(mapping.pairs())
        assert len(mapping.pairs()) == count
        assert execution_for(mapping.swm_id) == "DELEGATED_CPU"
        assert status_for(mapping.swm_id) == status
        if family == "Scene":
            assert set(mapping.swm_only) == {
                "cube.start_position",
                "cube.start_yaw",
                "cube.goal_position",
                "cube.goal_yaw",
            }
            before = env.get_state()
            with pytest.raises(ValueError):
                env.set_variation(**{"cube.start_yaw": [0.5]})
            assert all((env.get_state()[k] == v).all() for k, v in before.items(True, True))
