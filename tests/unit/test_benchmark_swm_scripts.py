"""Smoke runs of the stable-worldmodel comparison scripts: one env, B=1, 1 s windows, CPU only.

They run only where stable-worldmodel and the TwoRoom dependencies are installed.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.swm

pytest.importorskip("stable_worldmodel")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[2]
ENV = "TwoRoom"


def _run(module: str, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SDL_VIDEODRIVER": "dummy", "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


@pytest.mark.timeout(900)
def test_speed_writes_one_row_per_cell(tmp_path):
    from benchmarks.speed import COLUMNS

    out = tmp_path / "speed.csv"
    _run(
        "benchmarks.speed",
        "--cpu",
        "--step",
        "--render",
        "--env",
        ENV,
        "--batch",
        "1",
        "--actions-per-call",
        "1",
        "--window",
        "1",
        "--repeats",
        "1",
        "--repeats-small",
        "1",
        "--warmup-steps",
        "0",
        "--timed-steps",
        "0",
        "--length",
        "8",
        "--states-n",
        "16",
        "--cores",
        str(min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0),
        "--work",
        str(tmp_path / "work"),
        "--out",
        str(out),
    )
    columns, rows = _rows(out)
    assert columns == COLUMNS
    assert [(r["workload"], r["B"]) for r in rows] == [("step", "1"), ("render", "1")]
    for r in rows:
        assert r["status"] == "ok", r
        assert float(r["ours_rate"]) > 0 and float(r["source_rate"]) > 0
    assert out.with_suffix(".jsonl").exists()


@pytest.mark.timeout(900)
def test_render_mse_and_rollout_parity_write_their_columns(tmp_path):
    from benchmarks import render_mse, rollout_parity

    mse = tmp_path / "mse.csv"
    _run("benchmarks.render_mse", "--env", ENV, "--n", "4", "--work", str(tmp_path / "work"), "--out", str(mse))
    columns, rows = _rows(mse)
    assert columns == render_mse.COLUMNS and rows[0]["status"] == "ok", rows
    parity = tmp_path / "parity.csv"
    _run("benchmarks.rollout_parity", "--env", ENV, "--episodes", "1", "--steps", "3", "--out", str(parity))
    columns, rows = _rows(parity)
    assert columns == rollout_parity.COLUMNS and [r["mode"] for r in rows] == ["seeded", "natural"], rows
    assert all(r["status"] == "ok" for r in rows), rows
