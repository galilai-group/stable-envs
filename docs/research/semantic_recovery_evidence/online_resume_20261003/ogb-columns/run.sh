#!/bin/bash
#SBATCH --job-name=ogb-validation-columns
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen candidate source verified',len(manifest['files']),flush=True)
VERIFY
FAMILIES=(Cube Scene)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
lscpu
"$PY" -m pytest -q tests/unit/test_ogbench_validation_columns.py tests/unit/test_ogbench_manipulation_process.py tests/unit/test_ogbench_execution.py --junitxml="$BASE/tests-$FAMILY.xml"
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_validation_columns --domain "$FAMILY" --batches 64 128 --output "$BASE/results/$FAMILY"
"$PY" scripts/benchmarks/online_stepping/summarize_ogbench_validation_columns.py "$BASE/results/$FAMILY" "$BASE/tables/$FAMILY" --source "$PWD"
