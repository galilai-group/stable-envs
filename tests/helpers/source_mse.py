"""Recorded per-frame MSE against the source for each class, and the ratchet that checks a new measurement."""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).parents[1] / "fixtures" / "source_mse.json"
KEYS = ("mse_mean", "mse_p95", "mse_max")
SEED = 20260811
N = 64
RECORD = "scripts/testing/record_source_mse.py"

#: A value fails above ``RISE * recorded + SLACK`` and below ``DROP * recorded - SLACK``.
RISE = 1.10
DROP = 0.5
SLACK = 0.01

#: A declared ``visual_fidelity`` MSE agrees with the record to this relative or absolute margin.
DECLARED_REL = 0.05
DECLARED_ABS = 0.01


def load() -> dict:
    """``{class name: record}``; empty when the file does not exist."""
    return json.loads(PATH.read_text())["classes"] if PATH.exists() else {}


def record(name: str) -> dict | None:
    return load().get(name)


def ceiling(value: float) -> float:
    return RISE * value + SLACK


def problems(name: str, got: dict, rec: dict) -> list[str]:
    """Each statistic that rose above or dropped below its recorded band."""
    out = []
    for k in KEYS:
        have, want = float(got[k]), float(rec[k])
        if have > ceiling(want):
            out.append(f"{name} {k} = {have:.4f} rose above the record {want:.4f} (limit {ceiling(want):.4f})")
        elif have < DROP * want - SLACK:
            out.append(
                f"{name} {k} = {have:.4f} is far below the record {want:.4f}; if the improvement is intended, "
                f"record it with {RECORD} --update {name}"
            )
    return out


def declared_problems(name: str, vf: dict, rec: dict) -> list[str]:
    """Each ``visual_fidelity`` MSE that is missing or disagrees with the record."""
    out = []
    for k in KEYS:
        if k not in vf:
            out.append(f"{name}: visual_fidelity has no {k}; the record has {rec[k]:.4f}")
        elif abs(float(vf[k]) - float(rec[k])) > max(DECLARED_ABS, DECLARED_REL * float(rec[k])):
            out.append(f"{name}: visual_fidelity[{k!r}] = {vf[k]} but the record has {rec[k]:.4f}")
    return out
