"""The paper harness (benchmarks/paper/run.sh) end to end on CartPole, so a change of the public API breaks CI.

It runs the offline and frames+steps arms on CPU with 1 and 8 environments, ours and the source, with short
windows, and reads every output file back.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.swm,
    pytest.mark.skipif(
        not sys.platform.startswith("linux") or shutil.which("taskset") is None,
        reason="run.sh pins every arm to cores with taskset",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
ENV = "CartPole"


def _read(path: Path) -> dict:
    assert path.is_file(), f"run.sh did not write {path.name}"
    return json.loads(path.read_text())


def test_paper_harness_writes_every_cell_for_cartpole(tmp_path):
    out = tmp_path / "out"
    env = {
        **os.environ,
        "SNV_REPO": str(ROOT),
        "VENV_PYTHON": sys.executable,
        "STATES_DIR": str(tmp_path / "states"),
        "TMPDIR": str(tmp_path),
        "DEVICES": "cpu",
        "ARMS": "offline frames",
        "SMOKE": "1",
        "SWEEP_CPU": "1:8",
        "REPEATS": "1",
        "WINDOW": "0.2",
        "WARMUP": "0",
        "FRAME_CHECK": "0",
        "NPIN": "2",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "benchmarks" / "paper" / "run.sh"), str(out), f"classic={ENV}"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    run = _read(out / "run.json")
    procs = run["ncores"]

    offline = out / "offline" / ENV
    ours = _read(offline / "snv-cpu.json")["envs"][ENV]
    assert ours["input"] == "rows"
    points = ours["points"]
    assert [p["batch"] for p in points] == [1, 8]
    for p in points:
        assert p["status"] == "ok", p
        assert p["fps"] > 0 and p["frame_shape"] == [400, 600, 3]
    mse = _read(offline / "mse-cpu.json")["envs"][ENV]
    assert mse["status"] == "ok" and mse["input"] == "rows", mse
    assert mse["frames"] > 0 and mse["mse_mean"] >= 0
    for name in {"src-cpu-1p.json", f"src-cpu-{procs}p.json"}:
        cell = _read(offline / name)["envs"][ENV]
        assert cell["status"] == "ok", cell
        assert cell["fps"] > 0

    online = out / "online" / ENV
    for lanes in (1, 8):
        source = _read(online / f"src-render-cpu-L{lanes}.json")["arms"]["src-pipelined-render-cpu"]
        assert source["status"] == "ok", source
        assert source["steps_per_s"] > 0 and source["lanes"] == lanes
        ours = _read(online / f"snv-render-cpu-L{lanes}.json")["arms"]
        assert set(ours) == {"snv-render-cpu-T1", "snv-render-cpu-T16"}
        for arm in ours.values():
            assert arm["status"] == "ok", arm
            assert arm["steps_per_s"] > 0 and arm["lanes"] == lanes
