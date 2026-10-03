#!/bin/bash
#SBATCH --job-name=classic-parallel-source
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-parallel-source-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-parallel-source-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD" SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'candidate_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen source verified',len(manifest['files']),flush=True)
VERIFY
nvidia-smi
lscpu
DOMAINS=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
"$PY" scripts/benchmarks/online_stepping/classic_parallel_source_pipeline.py --domain "$DOMAIN" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --devices cpu cuda --batches 64 256 --workers 8 --output "$BASE/results/$DOMAIN.json"
"$PY" scripts/benchmarks/online_stepping/summarize_classic_parallel_source.py "$BASE/results/$DOMAIN.json" "$BASE/results/$DOMAIN.csv" --source "$PWD"
