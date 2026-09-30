"""Re-export the explicitly selected, validated evidence; performs no benchmarks."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[3]
inputs = json.loads((root / "export_inputs.json").read_text())
command = [sys.executable, str(repo / "scripts/benchmarks/renderer_table/export_paired_saved.py"), str(root), "--cube-gate", str(root / inputs["cube_gate"]), "--source-gpu-audit", str(root / inputs["source_gpu_audit"])]
for path in inputs["result_overlays"]:
    command += ["--result-overlay", str(root / path)]
subprocess.run(command, check=True)
