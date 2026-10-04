"""Verify final Rocket gates without changing any registration's status."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, terminal, output):
    assert not output.exists(), "Write a fresh verification directory."
    status = (root / "terminal.txt").read_text()
    assert status == "GPU_EXIT=0\nCPU_EXIT=0\n", status
    assert (root / "gpu-terminal.txt").read_text().strip() == "GPU_LANE_PASS"
    assert (root / "cpu-terminal.txt").read_text().strip() == "CPU_LANE_PASS"
    checks = {}
    for device in ("cpu", "cuda"):
        data = json.loads((root / f"public-{device}/result.json").read_text())
        job = str(data["job_id"])
        assert f"{job}|COMPLETED|0:0|" in terminal.read_text()
        assert data["device"] == device
        checks[device] = data
    assert checks["cpu"]["job_id"] == checks["cuda"]["job_id"]
    for name, expected in (("cpu-unit.log", 67), ("cuda-unit.log", 3)):
        text = (root / name).read_text()
        assert re.search(rf"\b{expected} passed\b", text), name
        assert not re.search(r"\b[1-9][0-9]* (?:failed|skipped|errors?)\b", text), name
    companions = []
    for name, expected, count, flag in [
        ("branch-source", "INDEPENDENT_SOURCE_SUFFIXES_PASS", 108, "source_numeric_and_future_pixels_exact"),
        ("persistent-variation", "PERSISTENT_VARIATION_PASS", 90, "source_reset_steps_pixels_restore_suffix_exact"),
    ]:
        data = json.loads((root / name / "result.json").read_text())
        assert data["status"] == expected and len(data["records"]) == count
        assert all(r[flag] for r in data["records"])
        for filename, digest in data["production_hashes"].items():
            assert sha(ROOT / filename) == digest, filename
        companions.append(dict(gate=name, cases=count))
    output.mkdir(parents=True)
    for device in ("cpu", "cuda"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_current_render.py")),
                "--input",
                str(root / f"public-{device}"),
                "--output",
                str(output / f"render-{device}.json"),
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_online_source.py")),
                "--input",
                str(root / f"online-{device}"),
                "--terminal",
                str(terminal),
                "--output",
                str(output / f"online-{device}.json"),
            ],
            check=True,
            cwd=ROOT,
        )
    result = dict(
        status="PASS_CURRENT_PRODUCTION_ROCKET_GATES",
        job=checks["cpu"]["job_id"],
        public_checkpoints_per_device=204,
        online_cases_per_device=64,
        online_frames_per_device=224,
        cpu_tests=67,
        cuda_tests=3,
        companions=companions,
        raw_pixel_verifications={
            name: json.loads((output / name).read_text())
            for name in ["render-cpu.json", "render-cuda.json", "online-cpu.json", "online-cuda.json"]
        },
        limitations=[
            "Acceptance requires matching source/runtime math profiles; cross-BLAS-profile exact replay is not promised.",
            "Bounded official-engine replay is O(history), through the registered task's 1,202-action horizon.",
            "No online speedup or all-family throughput acceptance is inferred from correctness.",
        ],
        verifier_sha256=sha(Path(__file__)),
        evidence_hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()},
        terminal_sha256=sha(terminal),
        promotion=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input, args.terminal, args.output)
    print(result["status"], result["job"])
