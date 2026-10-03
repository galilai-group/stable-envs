#!/bin/bash
#SBATCH --job-name=navigation-source
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/navigation-source-comparison-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/navigation-source-comparison-v1-20261003
EXTRAS=/oscar/scratch/sboughan/se-semantic-recovery/navigation-online-profile-v1-20261002/matplotlib3109
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD:$EXTRAS" MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
export MPLCONFIGDIR="$BASE/matplotlib/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys,matplotlib
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'candidate_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
source=json.loads((root/'source_manifest.json').read_text())
for rel,digest in source['files'].items():assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()==digest,rel
assert matplotlib.__version__=='3.10.9'
print('Frozen candidate/source verified',len(manifest['files']),len(source['files']),flush=True)
VERIFY
nvidia-smi
lscpu
DOMAINS=(two_room piecewise simple_point_maze)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
"$PY" scripts/benchmarks/online_stepping/navigation_source_pipeline.py --domain "$DOMAIN" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --devices cpu cuda --batches 16 64 --workers 8 --cpu-threads 8 --output "$BASE/results/$DOMAIN.json"
"$PY" scripts/benchmarks/online_stepping/summarize_navigation_source.py "$BASE/results/$DOMAIN.json" "$BASE/results/$DOMAIN.csv" --source "$PWD" --cpu-threads 8 --components-output "$BASE/results/$DOMAIN-components.csv"
