"""Machine-auditable replacement-readiness inventory.

Status is derived, not editorial:

    SYMBOLIC     no pixel cell
    INCOMPLETE   stable-envs currently only has enough capability to render
    PARTIAL      native or delegated step exists, but UNSUPPORTED_MEANINGFUL or PARTIAL leaves remain
    READY        execution is NATIVE or DELEGATED_CPU, and
                 UNSUPPORTED_MEANINGFUL = 0 and PARTIAL = 0

``READY`` is never assigned while a meaningful gap is listed beside it.
"""

from __future__ import annotations

from collections import Counter

from .crosswalk import SWM_IDS, SYMBOLIC_IDS, canonical_from_frozen
from .variation import VARIATION_MAPS

#: Historical SWM ids whose stable-envs port implements transition/step itself.
NATIVE_SWM_IDS = frozenset(
    {
        "swm/Piecewise-v0",
        "swm/SimplePointMaze-v0",
        "swm/TwoRoom-v1",
    }
)

# These registrations have a delegated default. Sibling task variants and the
# complete Variation surface are not certified by this architecture inventory;
# exact accepted scopes and evidence live in semantic_recovery_matrix.json.
DELEGATED_CPU_SWM_IDS = frozenset(
    {
        "swm/PFRocketLanding-v0",
        "swm/PushT-Discrete-v1",
        "swm/PushT-v1",
        "swm/BallInCupDMControl-v0",
        "swm/FingerDMControl-v0",
        "swm/ManipulatorDMControl-v0",
        "swm/QuadrupedDMControl-v0",
        "swm/AcrobotControl-v1",
        "swm/MountainCarControl-v0",
        "swm/MountainCarContinuousControl-v0",
        "swm/AcrobotDMControl-v0",
        "swm/CartPoleControl-v1",
        "swm/CartpoleDMControl-v0",
        "swm/CheetahDMControl-v0",
        "swm/HopperDMControl-v0",
        "swm/OGBMaze-v0",
        "swm/OGBCube-v0",
        "swm/OGBScene-v0",
        "swm/PendulumControl-v1",
        "swm/PendulumDMControl-v0",
        "swm/ReacherDMControl-v0",
        "swm/WalkerDMControl-v0",
    }
)

STATUSES = ("READY", "PARTIAL", "INCOMPLETE", "SYMBOLIC")
EXECUTIONS = ("NATIVE", "DELEGATED_CPU", "INCOMPLETE", "SYMBOLIC")


def execution_for(swm_id: str) -> str:
    if swm_id in SYMBOLIC_IDS:
        return "SYMBOLIC"
    if swm_id in NATIVE_SWM_IDS:
        return "NATIVE"
    if swm_id in DELEGATED_CPU_SWM_IDS:
        return "DELEGATED_CPU"
    return "INCOMPLETE"


def maps_for(swm_id: str) -> list:
    out = []
    for frozen in SWM_IDS.get(swm_id, ()):
        name = canonical_from_frozen(frozen)
        m = VARIATION_MAPS.get(name) or VARIATION_MAPS.get(frozen)
        if m is not None:
            out.append(m)
    return out


def merged_replacement_class(swm_id: str) -> dict[str, str]:
    """Per-leaf overlay across every class that serves ``swm_id``. NATIVE wins."""
    rank = {"NATIVE": 0, "PARTIAL": 1, "UNSUPPORTED_MEANINGFUL": 2, "DEAD_UPSTREAM": 3}
    merged: dict[str, str] = {}
    for m in maps_for(swm_id):
        for path, status in m.replacement_class().items():
            prev = merged.get(path)
            if prev is None or rank.get(status, 9) < rank.get(prev, 9):
                merged[path] = status
    return merged


def leaf_counts(swm_id: str) -> dict[str, int]:
    cls = merged_replacement_class(swm_id)
    return {
        "NATIVE": sum(v == "NATIVE" for v in cls.values()),
        "PARTIAL": sum(v == "PARTIAL" for v in cls.values()),
        "UNSUPPORTED_MEANINGFUL": sum(v == "UNSUPPORTED_MEANINGFUL" for v in cls.values()),
        "DEAD_UPSTREAM": sum(v == "DEAD_UPSTREAM" for v in cls.values()),
    }


def remaining_leaves(swm_id: str, status: str) -> tuple[str, ...]:
    cls = merged_replacement_class(swm_id)
    return tuple(sorted(p for p, s in cls.items() if s == status))


def status_for(swm_id: str) -> str:
    exe = execution_for(swm_id)
    if exe == "SYMBOLIC":
        return "SYMBOLIC"
    if exe == "INCOMPLETE":
        return "INCOMPLETE"
    counts = leaf_counts(swm_id)
    if counts["UNSUPPORTED_MEANINGFUL"] == 0 and counts["PARTIAL"] == 0:
        return "READY"
    return "PARTIAL"


def inventory() -> list[dict]:
    rows = []
    for swm_id in SWM_IDS:
        exe = execution_for(swm_id)
        counts = leaf_counts(swm_id)
        rows.append(
            {
                "swm_id": swm_id,
                "classes": tuple(canonical_from_frozen(c) for c in SWM_IDS[swm_id]),
                "execution": exe,
                "status": status_for(swm_id),
                "unsupported_meaningful": remaining_leaves(swm_id, "UNSUPPORTED_MEANINGFUL"),
                "partial": remaining_leaves(swm_id, "PARTIAL"),
                "counts": counts,
            }
        )
    return rows


def status_counts() -> dict[str, int]:
    return dict(Counter(status_for(sid) for sid in SWM_IDS))


def execution_counts() -> dict[str, int]:
    return dict(Counter(execution_for(sid) for sid in SWM_IDS))


def all_remaining(status: str) -> dict[str, tuple[str, ...]]:
    return {sid: remaining_leaves(sid, status) for sid in SWM_IDS if remaining_leaves(sid, status)}
