#!/bin/bash
#SBATCH --job-name=fetch-source-pipeline
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/fetch-source-pipeline-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-source-pipeline-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name, prefix in [('candidate_manifest.json','code'),('source_manifest.json','source')]:
    manifest=json.loads((root/name).read_text())
    for rel,digest in manifest['files'].items():
        assert hashlib.sha256((root/prefix/rel).read_bytes()).hexdigest()==digest,rel
    print(name, len(manifest['files']), 'verified', flush=True)
VERIFY
nvidia-smi
lscpu
FAMILIES=(swm/FetchReach-v3 swm/FetchReachDense-v3 swm/FetchReachDict-v3 swm/FetchPush-v3 swm/FetchPushDense-v3 swm/FetchPushDict-v3 swm/FetchSlide-v3 swm/FetchSlideDense-v3 swm/FetchSlideDict-v3 swm/FetchPickAndPlace-v3 swm/FetchPickAndPlaceDense-v3 swm/FetchPickAndPlaceDict-v3)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_NAME=${FAMILY#swm/}
"$PY" scripts/benchmarks/online_stepping/fetch_source_oscar.py --batches 64 --order-offset "$SLURM_ARRAY_TASK_ID" --registration "$FAMILY" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --output "$BASE/results/$OUTPUT_NAME"

"$PY" -m scripts.benchmarks.online_stepping.summarize_fetch_source --results "$BASE/results/$OUTPUT_NAME" --candidate "$BASE/code" --source "$BASE/source" --source-manifest "$BASE/source_manifest.json" --batches 64 --output "$BASE/summary/$OUTPUT_NAME"
