"""Explicit case accounting, independent of optional runtime registration imports.

These are gate requirements, not a new acceptance matrix. A sibling's result can
never satisfy a registration's case. Family witnesses remain in their drivers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class CaseKey:
    registration: str
    axis: str
    scenario: str
    device: str = "cpu"


def coverage(required, results) -> dict:
    """Return PASS only for the exact nonempty required case set, all passing.

    Results are (CaseKey, status) pairs so duplicates cannot be silently lost in
    dictionary construction. Missing/skipped/NA required cases block the gate.
    """
    required, results = list(required), list(results)
    if not required or len(set(required)) != len(required):
        raise ValueError("Requirements must be unique and nonempty")
    seen, duplicates, invalid = {}, [], []
    for key, status in results:
        if key in seen:
            duplicates.append(key)
        if status not in {"PASS", "FAIL", "BLOCKED", "SKIPPED", "NOT_APPLICABLE"}:
            invalid.append((key, status))
        seen[key] = status
    missing = set(required) - seen.keys()
    unexpected = seen.keys() - set(required)
    nonpassing = {key: value for key, value in seen.items() if value != "PASS"}
    status = "PASS"
    if duplicates or unexpected or invalid or "FAIL" in seen.values():
        status = "FAIL"
    elif missing or nonpassing:
        status = "BLOCKED"
    return dict(
        status=status,
        required=len(required),
        passed=sum(seen.get(k) == "PASS" for k in required),
        missing=sorted(missing),
        unexpected=sorted(unexpected),
        duplicates=duplicates,
        nonpassing=nonpassing,
        invalid=invalid,
    )


_CLASSIC = "tests/integration/test_native_classic_gate.py::test_frozen_reference_oracle"
_CLASSIC_REST = "tests/integration/test_native_classic_remaining_gate.py::test_frozen_remaining_reference_oracle"
_NAVIGATION = (
    "tests/integration/test_native_navigation_oracle.py::test_frozen_native_navigation_oracles_replay_exactly"
)
_DMC_FIRST = "tests/integration/test_dmc_first_family_semantics.py::test_first_dmc_family_full_state_semantics"
_DMC_ALIGNED = "tests/integration/test_dmc_aligned_semantics.py::test_aligned_full_state_specs_and_gate"
_DMC_CONTINUATION = "tests/integration/test_dmc_continuation_semantics.py::test_continuation_dmc_full_state_semantics"
_DMC_SIBLING = "tests/unit/test_dmc_sibling_tasks.py::test_cell_matches_the_source_task_and_restores_into_a_fresh_env"
_FETCH = "tests/integration/test_fetch_variation.py::test_reset_leaves_match_swm"
_OGB = "tests/integration/test_ogb_variation.py::test_reset_option_leaves_match_swm_per_lane"
_MAZE = "tests/integration/test_ogbench_semantics.py::test_every_maze_cell_matches_the_official_env"
_CRAFTAX = "tests/unit/test_craftax_semantics.py::test_source_reset_all_actions_and_reset_replay"
_PUSHT = "tests/integration/test_variation_pusht.py::test_leaf_matches_swm_per_lane"

#: Registered env -> the CPU test comparing it with its source (live, or a corpus recorded from it).
#: tests/unit/test_registry_coverage.py requires one per registered env unless tests/exemptions.toml waives it.
SOURCE_GATES: dict[str, str] = {
    "CartPole": _CLASSIC,
    "Pendulum": _CLASSIC,
    "Acrobot": _CLASSIC_REST,
    "MountainCar": _CLASSIC_REST,
    "MountainCarContinuous": _CLASSIC_REST,
    "TwoRoom": _NAVIGATION,
    "Piecewise": _NAVIGATION,
    "SimplePointMaze": _NAVIGATION,
    "PiecewiseZones": "tests/integration/test_variation_navigation.py::test_piecewise_target_and_bias_field_match_swm",
    "PushT": _PUSHT,
    "PushTDiscrete": _PUSHT,
    "DMCAcrobotBalance": _DMC_FIRST,
    "DMCPendulumSwingup": _DMC_FIRST,
    "DMCCartpoleSwingup": _DMC_FIRST,
    "DMCBallInCupCatch": _DMC_ALIGNED,
    "DMCFingerTurnHard": _DMC_ALIGNED,
    "DMCManipulatorBringBall": _DMC_ALIGNED,
    "DMCQuadrupedWalk": _DMC_ALIGNED,
    "DMCCheetahRun": _DMC_CONTINUATION,
    "DMCHopperHop": _DMC_CONTINUATION,
    "DMCWalkerWalk": _DMC_CONTINUATION,
    "DMCHumanoidWalk": "tests/integration/test_dmc_humanoid_semantics.py::test_humanoid_walk_identity_specs_and_full_state_semantics",
    "DMCReacherHard": "tests/integration/test_reacher_semantics.py::test_reacher_full_state_semantics",
    **dict.fromkeys(
        (
            "DMCCartpoleBalance",
            "DMCFingerSpin",
            "DMCFingerTurnEasy",
            "DMCPointMassEasy",
            "DMCReacherEasy",
            "DMCSwimmerSwimmer6",
        ),
        _DMC_SIBLING,
    ),
    **{f"Fetch{task}": _FETCH for task in ("Reach", "Push", "Slide", "PickAndPlace")},
    "OGBCube": _OGB,
    "OGBScene": _OGB,
    **{
        f"{agent}Maze{size}{task}": _MAZE
        for agent, sizes, tasks in (
            ("Point", ("Medium", "Large", "Giant", "Teleport"), ("Navigate", "Stitch")),
            ("Ant", ("Medium", "Large", "Giant", "Teleport"), ("Navigate", "Stitch")),
            ("Ant", ("Medium", "Large", "Teleport"), ("Explore",)),
            ("Humanoid", ("Medium", "Large", "Giant"), ("Navigate", "Stitch")),
        )
        for size in sizes
        for task in tasks
    },
    **dict.fromkeys(("CraftaxClassicPixels", "CraftaxClassicSymbolic", "CraftaxPixels", "CraftaxSymbolic"), _CRAFTAX),
    "PyFlytRocketLanding": "tests/integration/test_rocket_source_pixels.py::test_source_rollout_pixels",
}
