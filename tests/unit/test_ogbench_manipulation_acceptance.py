"""The recovered public configuration must agree with its bounded readiness record."""

import json
from pathlib import Path

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import execution_for, status_for

ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize("family,cases,cpu", [("Cube", 13, 45), ("Scene", 22, 48)])
def test_each_family_has_its_own_complete_acceptance_evidence(family, cases, cpu):
    evidence = ROOT / "docs/research/semantic_recovery_evidence/ogb_renderer_retry"
    verified = json.loads((evidence / "verified-gpu.json").read_text())
    row = verified["families"][family]
    assert verified["job"] == 6669909 and row["status"] == "GPU_EVIDENCE_VERIFIED"
    assert row["frame_cases"] == cases and row["cpu_tests"] == cpu
    assert all(c["full_render_baseline_exact"] for c in row["rows"])
    assert row["resource_lifetime"]["context_freed"] == 0
    assert all(row["variations"].values())
    original = evidence / "remote/evidence" / family
    assert (original / "cpu.xml").is_file()
    assert f"OGB_{family.upper()}_GPU_GATE_PASS" in (original / "gpu.log").read_text().splitlines()
