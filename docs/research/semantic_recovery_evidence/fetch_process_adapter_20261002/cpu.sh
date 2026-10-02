#!/bin/bash
set -euo pipefail
cd /oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH="$PWD"
export XDG_CACHE_HOME=/oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/cache
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
"$PY" -m pytest -q tests/unit/test_fetch_process_adapter.py tests/unit/test_step_capture.py
"$PY" scripts/benchmarks/online_stepping/fetch_process_adapter.py --output /oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/cpu.json
