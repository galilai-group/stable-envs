"""Validate the bounded CPU reports and aggregate per-frame MSE statistics."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
candidate = json.loads((root / "candidate_manifest.json").read_text())["files"]
source = json.loads((root / "source_manifest.json").read_text())["files"]
scheduler = (root.parent / "cluster-status.txt").read_text()
rows, artifacts = [], {}
for index, family in enumerate(("Cube", "Scene"), 1):
    path = root / "terminal/6991509" / family / "cpu.json"
    report = json.loads(path.read_text())
    assert report["status"] == "PASS" and report["family"] == family and report["device"] == "cpu"
    assert report["batch"] == report["workers"] == 2 and report["no_timing"]
    assert report["source_files"] == source
    assert f"6991509_{index}|COMPLETED|0:0|" in scheduler
    for name, digest in report["candidate_files"].items():
        assert candidate[name] == digest, name
    assert len(report["cases"]) == 30 and len(report["worker_closures"]) == 20
    assert {(c["horizon"], c["selection"], c["reached_lanes"]) for c in report["cases"]} == set(
        itertools.product((1, 8), ("none", "empty", "final", "all", "reverse_sparse"), (0, 1, 2)))
    for case in report["cases"]:
        for name in ("source_worker_exact", "source_engine_rng_witness", "source_render_nonmutation",
                     "public_state_ownership", "frame_order_validity_and_lifecycle", "initial_state_roundtrip"):
            assert case[name] == "PASS"
        for identity in (case["source_gl"], *case["worker_gl"]):
            assert identity["software"] and "llvmpipe" in identity["renderer"]
            assert identity["source_render_gl_errors"] == [1281]
            assert identity["source_light_cutoffs"] == [45.0, 100.0]
        assert case["mse_max"] is None or 0 <= case["mse_max"] <= .001
    for closure in report["worker_closures"]:
        assert all(not worker["alive"] and worker["exitcode"] == 0 for worker in closure["workers"])
    measured = [c for c in report["cases"] if c["frames_checked"]]
    count = sum(c["frames_checked"] for c in measured)
    mean = sum(c["frames_checked"] * c["mse_mean"] for c in measured) / count
    variance = sum((c["frames_checked"] - 1) * (c["mse_sd"] or 0) ** 2
                   + c["frames_checked"] * (c["mse_mean"] - mean) ** 2 for c in measured) / (count - 1)
    rows.append(dict(family=family, device="cpu", cases=30, frames=count, mse_mean=mean, mse_sd=math.sqrt(variance),
                     mse_max=max(c["mse_max"] for c in measured), status="PASS_BOUNDED_CPU_FRAME_GATE"))
    artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(status="PASS_BOUNDED_CPU_FRAME_GATES", scope="B2/T1,T8 software-GL only; no CUDA or speedup acceptance",
              rows=rows, reports=artifacts,
              teardown_note="All 20 recorded worker closures per family are clean. Logs retain MuJoCo interpreter-finalization GLFW destructor warnings; these are not suppressed.")
output = root / "cpu-verification.json"
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(result, indent=2) + "\n")
with (root / "cpu-mse.csv").open("x") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(result, indent=2))
