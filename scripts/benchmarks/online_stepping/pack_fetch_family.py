"""Build a self-contained, hash-recorded Fetch benchmark upload from this checkout."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    paths = subprocess.check_output(
        ["git", "ls-files", "stable_envs", "tests/helpers", "tests/__init__.py"], cwd=root, text=True
    ).splitlines()
    paths += [
        "pyproject.toml",
        "tests/conftest.py",
        "tests/fixtures/regression/support_matrix.json",
        "tests/unit/test_fetch_process_adapter.py",
        "tests/unit/test_step_capture.py",
        "scripts/benchmarks/online_stepping/fetch_family.py",
        "scripts/benchmarks/online_stepping/fetch_process_adapter.py",
    ]
    paths = sorted(set(paths))
    archive = args.output / "code.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in paths:
            tar.add(root / name, arcname=name, recursive=False)
    with tarfile.open(archive) as tar:
        tar.extractall(args.output / "code", filter="data")
    manifest = dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths},
    )
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # The remote base is an explicit controlled absolute path, not arbitrary shell input.
    if not args.remote.startswith("/oscar/scratch/") or any(c not in "/-_." and not c.isalnum() for c in args.remote):
        raise ValueError("Expected an Oscar scratch path containing only simple path characters")
    (args.output / "run.sh").write_text(f"""#!/bin/bash
set -euo pipefail
BASE={args.remote}
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
TASKS=(Reach Push Slide PickAndPlace)
nvidia-smi
lscpu
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    "$PY" -m pytest -q -rs tests/unit/test_fetch_process_adapter.py tests/unit/test_step_capture.py
fi
"$PY" scripts/benchmarks/online_stepping/fetch_family.py --task "${{TASKS[$SLURM_ARRAY_TASK_ID]}}" --output "$BASE/results/${{TASKS[$SLURM_ARRAY_TASK_ID]}}.json"
""")
    print(args.output)


if __name__ == "__main__":
    main()
