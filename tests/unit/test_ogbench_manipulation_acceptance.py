"""The recovered public Cube/Scene configuration must agree with its readiness record."""

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import execution_for, status_for


@pytest.mark.parametrize("family,count", [("Cube", 11), ("Scene", 12)])
def test_live_recovered_configuration_matches_readiness(family, count):
    pytest.importorskip("ogbench")
    from stable_envs.envs.ogbench import OGBCube, OGBScene

    cls = OGBCube if family == "Cube" else OGBScene
    with cls(seed=17) as env:
        mapping = VARIATION_MAPS[cls.__name__]
        assert set(env.variation_space.names()) == set(mapping.pairs())
        assert len(mapping.pairs()) == count
        assert execution_for(mapping.swm_id) == "DELEGATED_CPU"
        assert status_for(mapping.swm_id) == "READY"
