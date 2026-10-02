#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-pipeline-single-action-20261002
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache"
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
nvidia-smi
lscpu
for VERSION in v2 v3; do
    cd "/oscar/scratch/sboughan/se-semantic-recovery/fetch-pipeline-production-$VERSION-20261002/code"
    export PYTHONPATH="$PWD"
    "$PY" scripts/benchmarks/online_stepping/fetch_pipeline.py --task Slide --batches 64 --horizons 1 --repeats 10 --blocks 4 --suffixes plain --production --output "$BASE/$VERSION.json"
done
