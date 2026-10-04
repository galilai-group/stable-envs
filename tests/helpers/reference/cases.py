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
