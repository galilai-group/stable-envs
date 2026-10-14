"""Applicability readers in ``stable_envs.testing.conformance`` and the Rocket colour declaration."""

from pathlib import Path

import numpy as np
import pytest

from stable_envs.testing.conformance import (
    color_exactness_applicability,
    partition_colour_exactness,
    resolution_consistency_applicability,
)

ROOT = Path(__file__).resolve().parents[2]

#: The five leaves PFRocketLanding declares, and nothing else: the declaration is scoped by name.
ROCKET_LEAVES = ("pad.color", "rocket.body_color", "rocket.booster_color", "rocket.fin_color", "rocket.leg_color")


def test_exclusion_only_applies_to_all_states_outside_exclusion():
    class Env:
        state_names = ["light"]
        resolution_consistency_applicability = {"light < 0.5": "not_applicable: night"}

    result = resolution_consistency_applicability(Env(), np.array([[0.5], [1.0]]))
    assert result["applicable"] is True
    assert result["indices"] is None
    result = resolution_consistency_applicability(Env(), np.array([[0.2], [0.8]]))
    assert result["indices"] == [1]


def test_empty_explicit_inclusion_does_not_invent_complement():
    class Env:
        state_names = ["light"]
        resolution_consistency_applicability = {"light >= 0.8": "applicable", "light < 0.2": "not_applicable: dark"}

    result = resolution_consistency_applicability(Env(), np.array([[0.1], [0.5]]))
    assert result["applicable"] is False
    assert result["indices"] == []


def test_no_colour_declaration_leaves_every_leaf_judged():
    class Env:
        pass

    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])
    assert result["honoured"] is False
    assert result["not_applicable_probes"] == []
    assert result["undeclared_leaves_still_judged"] == ["a.color->[17, 33, 200]"]


@pytest.mark.parametrize(
    "missing", ["leaves", "mechanism", "max_shading_factor", "official_renderer_fails_the_same_check", "evidence"]
)
def test_a_declaration_missing_its_evidence_is_refused(missing):
    """Every required key is load-bearing: drop any one and the check stays judged."""
    full = {
        "leaves": ["a.color"],
        "mechanism": "a law",
        "max_shading_factor": 0.99,
        "official_renderer_fails_the_same_check": True,
        "evidence": ["somewhere.json"],
    }
    Env = type("Env", (), {"color_exactness_applicability": {k: v for k, v in full.items() if k != missing}})
    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])
    assert result["honoured"] is False
    assert missing in result["rejected_because"]


def test_a_law_that_reaches_unit_gain_is_refused():
    """The whole argument is that the ceiling is below 1. A ceiling of 1 has no case to make."""
    Env = type(
        "Env",
        (),
        {
            "color_exactness_applicability": {
                "leaves": ["a.color"],
                "mechanism": "a law",
                "max_shading_factor": 1.0,
                "official_renderer_fails_the_same_check": True,
                "evidence": ["somewhere.json"],
            }
        },
    )
    assert color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])["honoured"] is False


def test_an_undeclared_leaf_is_not_covered_by_a_declared_one():
    Env = type(
        "Env",
        (),
        {
            "color_exactness_applicability": {
                "leaves": ["a.color"],
                "mechanism": "a law",
                "max_shading_factor": 0.9,
                "official_renderer_fails_the_same_check": True,
                "evidence": ["somewhere.json"],
            }
        },
    )
    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]", "b.color->[17, 33, 200]"])
    assert result["honoured"] is True
    assert result["not_applicable_probes"] == ["a.color->[17, 33, 200]"]
    assert result["undeclared_leaves_still_judged"] == ["b.color"]


def test_the_rocket_declares_exactly_the_five_lit_leaves():
    """Read off the CLASS, with no env constructed and no PyFlyt installed."""
    from stable_envs.envs.rocket.legacy import PyFlytRocketLanding

    decl = PyFlytRocketLanding.color_exactness_applicability
    assert tuple(decl["leaves"]) == ROCKET_LEAVES
    assert decl["max_shading_factor"] == 0.993567
    assert decl["official_renderer_fails_the_same_check"] is True
    assert any("colour_law_ceiling.json" in e for e in decl["evidence"])
    for cited in decl["evidence"]:
        assert (ROOT / cited.split(":")[0]).exists(), cited


def _probe(leaf, target="[17, 33, 200]"):
    return f"{leaf}->{target}"


def test_partition_without_a_declaration_asserts_everything():
    """The previous behaviour, exactly: no declaration means every failure is asserted on."""
    part = partition_colour_exactness({"colors_not_applied": [_probe("a.color"), _probe("b.color")]})
    assert part["honoured"] is False
    assert part["asserted"] == [_probe("a.color"), _probe("b.color")]
    assert part["not_applicable"] == []


def test_partition_asserts_the_undeclared_leaves_and_only_those():
    result = {
        "colors_not_applied": [_probe("a.color"), _probe("b.color")],
        "best_abs_diff": {_probe("a.color"): 4, _probe("b.color"): 9},
        "applicability": {"honoured": True, "why": "a law", "not_applicable_probes": [_probe("a.color")]},
    }
    part = partition_colour_exactness(result)
    assert part["asserted"] == [_probe("b.color")]
    assert part["not_applicable"] == [{"probe": _probe("a.color"), "best_abs_diff": 4}]


def test_partition_keeps_the_measurement_beside_the_excused_leaf():
    """The declaration stops the assertion, never the measurement: the number has to come through."""
    result = {
        "colors_not_applied": [_probe("a.color")],
        "best_abs_diff": {_probe("a.color"): 7},
        "applicability": {"honoured": True, "not_applicable_probes": [_probe("a.color")]},
    }
    part = partition_colour_exactness(result)
    assert part["asserted"] == []
    assert part["not_applicable"][0]["best_abs_diff"] == 7


def test_partition_on_a_passing_env_is_empty_either_way():
    for result in ({"colors_not_applied": []}, {"colors_not_applied": [], "applicability": {"honoured": True}}):
        part = partition_colour_exactness(result)
        assert part["asserted"] == [] and part["not_applicable"] == []


def test_the_rocket_declaration_drives_the_partition_end_to_end():
    """The class declaration, the real probe names, and the helper the GPU suite calls."""
    from stable_envs.envs.rocket.legacy import PyFlytRocketLanding

    probes = [_probe(leaf, t) for leaf in ROCKET_LEAVES for t in ("[17, 33, 200]", "[200, 17, 33]")]
    result = {
        "colors_not_applied": probes + [_probe("environment.sky_color")],
        "best_abs_diff": {p: 3 for p in probes + [_probe("environment.sky_color")]},
        "applicability": color_exactness_applicability(
            PyFlytRocketLanding, probes + [_probe("environment.sky_color")]
        ),
    }
    part = partition_colour_exactness(result)
    # The five declared leaves are excused; a sixth leaf nobody declared still fails.
    assert part["asserted"] == [_probe("environment.sky_color")]
    assert len(part["not_applicable"]) == len(probes)
