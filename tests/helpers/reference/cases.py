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
_DMC_SUITE = "tests/unit/test_dmc_sibling_tasks.py::test_cell_matches_the_source_task_and_restores_into_a_fresh_env"
_DMC_HISTORICAL = "tests/unit/test_dmc_sibling_tasks.py::test_historical_cell_matches_the_source_task"
_DMC_CORPUS = "tests/unit/test_dmc_oracle_replay.py::test_adapter_replays_every_recorded_step_from_restored_states"
_MAZE = "tests/integration/test_ogbench_semantics.py::test_every_maze_cell_matches_the_official_env"
_MAZES = tuple(
    f"{agent}Maze{size}{task}"
    for agent, sizes, tasks in (
        ("Point", ("Medium", "Large", "Giant", "Teleport"), ("Navigate", "Stitch")),
        ("Ant", ("Medium", "Large", "Giant", "Teleport"), ("Navigate", "Stitch")),
        ("Ant", ("Medium", "Large", "Teleport"), ("Explore",)),
        ("Humanoid", ("Medium", "Large", "Giant"), ("Navigate", "Stitch")),
    )
    for size in sizes
    for task in tasks
)
_CRAFTAX = "tests/unit/test_craftax_semantics.py::test_source_reset_all_actions_and_reset_replay"

#: Registered env -> the CPU test comparing it with the library it ports, or with a corpus recorded from
#: its source. It must not import stable_worldmodel: this layer stays when the SWM layer is removed.
ORIGINAL_GATES: dict[str, str] = {
    **dict.fromkeys(("CartPole", "Pendulum"), _CLASSIC),
    **dict.fromkeys(("Acrobot", "MountainCar", "MountainCarContinuous"), _CLASSIC_REST),
    **dict.fromkeys(("TwoRoom", "Piecewise", "SimplePointMaze"), _NAVIGATION),
    **dict.fromkeys(
        (
            "DMCAcrobotBalance",
            "DMCPendulumSwingup",
            "DMCCartpoleSwingup",
            "DMCCheetahRun",
            "DMCHopperHop",
            "DMCHumanoidWalk",
            "DMCReacherHard",
        ),
        _DMC_HISTORICAL,
    ),
    **dict.fromkeys(
        ("DMCBallInCupCatch", "DMCFingerTurnHard", "DMCManipulatorBringBall", "DMCQuadrupedWalk", "DMCWalkerWalk"),
        _DMC_CORPUS,
    ),
    **dict.fromkeys(
        (
            "DMCCartpoleBalance",
            "DMCFingerSpin",
            "DMCFingerTurnEasy",
            "DMCPointMassEasy",
            "DMCReacherEasy",
            "DMCSwimmerSwimmer6",
        ),
        _DMC_SUITE,
    ),
    **{
        f"Fetch{task}": "tests/integration/test_fetch_source.py::test_default_episode_matches_gymnasium_robotics"
        for task in ("Reach", "Push", "Slide", "PickAndPlace")
    },
    **dict.fromkeys(_MAZES, _MAZE),
    **dict.fromkeys(("CraftaxClassicPixels", "CraftaxClassicSymbolic", "CraftaxPixels", "CraftaxSymbolic"), _CRAFTAX),
}

_DMC_FIRST = "tests/integration/test_dmc_first_family_semantics.py::test_first_dmc_family_full_state_semantics"
_DMC_ALIGNED = "tests/integration/test_dmc_aligned_semantics.py::test_aligned_full_state_specs_and_gate"
_DMC_CONTINUATION = "tests/integration/test_dmc_continuation_semantics.py::test_continuation_dmc_full_state_semantics"
_PUSHT = "tests/integration/test_variation_pusht.py::test_leaf_matches_swm_per_lane"

#: Registered env with a stable-worldmodel registration -> the CPU test comparing it with live SWM
#: (marked ``swm``). Removing the SWM layer deletes this table with the ``swm`` tests.
SWM_GATES: dict[str, str] = {
    **dict.fromkeys(
        ("CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"),
        "tests/integration/test_classic_variation.py::test_classic_physics_matches_swm",
    ),
    **dict.fromkeys(
        ("TwoRoom", "Piecewise", "SimplePointMaze"),
        "tests/integration/test_native_navigation_gate.py::test_native_navigation_cpu_gate",
    ),
    "PiecewiseZones": "tests/integration/test_variation_navigation.py::test_piecewise_target_and_bias_field_match_swm",
    **dict.fromkeys(("PushT", "PushTDiscrete"), _PUSHT),
    **dict.fromkeys(("DMCAcrobotBalance", "DMCPendulumSwingup", "DMCCartpoleSwingup"), _DMC_FIRST),
    "DMCCartpoleBalance": "tests/integration/test_dmc_physics_variation.py::test_physics_leaves_match_swm",
    **dict.fromkeys(
        ("DMCBallInCupCatch", "DMCFingerTurnHard", "DMCManipulatorBringBall", "DMCQuadrupedWalk"), _DMC_ALIGNED
    ),
    **dict.fromkeys(("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk"), _DMC_CONTINUATION),
    "DMCHumanoidWalk": "tests/integration/test_dmc_humanoid_semantics.py::test_humanoid_walk_identity_specs_and_full_state_semantics",
    "DMCReacherHard": "tests/integration/test_reacher_semantics.py::test_reacher_full_state_semantics",
    **{
        f"Fetch{task}": "tests/integration/test_fetch_variation.py::test_reset_leaves_match_swm"
        for task in ("Reach", "Push", "Slide", "PickAndPlace")
    },
    **dict.fromkeys(
        ("OGBCube", "OGBScene"), "tests/integration/test_ogb_variation.py::test_reset_option_leaves_match_swm_per_lane"
    ),
    **dict.fromkeys(
        ("PointMazeLargeNavigate", "PointMazeLargeStitch"),
        "tests/integration/test_ogbench_semantics.py::test_ogbench_maze_full_state_semantics",
    ),
    "PyFlytRocketLanding": "tests/integration/test_rocket_source_pixels.py::test_source_rollout_pixels",
}
