#!/bin/bash
#SBATCH --job-name=classic-snapshot
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME" "$BASE/results"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():
 assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen candidate manifest verified',len(manifest['files']),flush=True)
VERIFY
DOMAINS=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
lscpu
"$PY" -m pytest -q tests/unit/test_classic_execution.py tests/unit/test_classic_state_export.py tests/unit/test_native_classic_semantics.py tests/unit/test_native_classic_remaining_semantics.py tests/unit/test_acrobot_process.py tests/unit/test_classic_snapshot.py --junitxml="$BASE/results/$DOMAIN-tests.xml"
"$PY" scripts/benchmarks/online_stepping/classic_snapshot.py --domain "$DOMAIN" --output "$BASE/results/$DOMAIN.json"
"$PY" scripts/benchmarks/online_stepping/summarize_classic_snapshot.py "$BASE/results/$DOMAIN.json" "$BASE/results/$DOMAIN.csv" --source "$PWD"
