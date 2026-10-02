#!/bin/bash
set -euo pipefail
cd /oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH="$PWD"
export XDG_CACHE_HOME=/oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/cache
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
nvidia-smi
"$PY" scripts/benchmarks/online_stepping/fetch_process_adapter.py --device cuda --gate-only --output /oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/cuda-gates.json
"$PY" scripts/benchmarks/online_stepping/fetch_process_adapter.py --device cuda --output /oscar/scratch/sboughan/se-semantic-recovery/fetch-process-adapter-20261002/cuda-bench.json --batches 64 256 1024 --blocks 1 --repeats 5
