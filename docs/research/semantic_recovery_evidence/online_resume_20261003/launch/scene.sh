#!/bin/bash
#SBATCH --job-name=scene-columns-cuda-retry
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-scene-cuda-v2-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-scene-cuda-v2-20261003
FROZEN=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
cd "$FROZEN/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_JOB_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$FROZEN" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen candidate source verified',len(manifest['files']),flush=True)
VERIFY
nvidia-smi
lscpu
"$PY" -m pytest -q tests/unit/test_ogbench_validation_columns.py tests/unit/test_ogbench_manipulation_process.py tests/unit/test_ogbench_execution.py --junitxml="$BASE/tests-Scene.xml"
"$PY" scripts/benchmarks/online_stepping/ogbench_validation_columns.py --domain Scene --devices cuda --batches 64 128 --verify-gl --output "$BASE/cuda.json"
