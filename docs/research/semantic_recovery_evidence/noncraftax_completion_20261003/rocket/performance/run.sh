#!/bin/bash
#SBATCH --job-name=rocket-replay-cost
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-replay-cost-v1-20261003/cost-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/rocket-replay-cost-v1-20261003
ORIGINAL=/oscar/scratch/sboughan/se-semantic-recovery/rocket-current-gate-v1-20261003
RUN="$BASE/results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN/cache"
cd "$ORIGINAL/code"
export PYTHONPATH="$PWD:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$RUN/cache" TORCH_EXTENSIONS_DIR="$RUN/cache/torch" CUDA_CACHE_PATH="$RUN/cache/cuda"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
"$PY" - "$BASE" "$ORIGINAL" <<'VERIFY'
import hashlib,json,pathlib,sys
base,original=map(pathlib.Path,sys.argv[1:])
for rel,digest in json.loads((original/'candidate_manifest.json').read_text())['files'].items():
    assert hashlib.sha256((original/'code'/rel).read_bytes()).hexdigest()==digest,rel
script=json.loads((base/'script-manifest.json').read_text())
assert hashlib.sha256((base/script['path']).read_bytes()).hexdigest()==script['sha256']
print('Frozen correctness candidate and separate cost harness verified',flush=True)
VERIFY
nvidia-smi > "$RUN/nvidia-smi.txt"
lscpu > "$RUN/lscpu.txt"
MODES=(--render-cpu --render-cpu --gpu --gpu)
BATCHES=(1 16 1 16)
"$PY" "$BASE/scripts/research_rocket_production/replay_performance.py" "${MODES[$SLURM_ARRAY_TASK_ID]}" --batches "${BATCHES[$SLURM_ARRAY_TASK_ID]}" --output "$RUN/performance" > "$RUN/performance.log" 2>&1
