#!/bin/bash
#SBATCH --job-name=classic-source-pipeline
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
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-source-pipeline-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-source-pipeline-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD" SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
DOMAINS=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/classic_source_pipeline.py --domain "$DOMAIN" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --batches 1 64 256 --output "$BASE/results/$DOMAIN.json"
"$PY" scripts/benchmarks/online_stepping/summarize_classic_source_pipeline.py "$BASE/results/$DOMAIN.json" "$BASE/results/$DOMAIN.csv" --source "$PWD"
