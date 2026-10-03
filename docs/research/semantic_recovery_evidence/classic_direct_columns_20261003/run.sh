#!/bin/bash
#SBATCH --job-name=classic-direct-columns
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --array=0-4
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-direct-columns-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-direct-columns-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
nvidia-smi
lscpu
DOMAINS=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
"$PY" -m pytest -q tests/unit/test_classic_state_export.py -k "$DOMAIN" --junitxml="$BASE/tests-$DOMAIN.xml"
"$PY" scripts/benchmarks/online_stepping/classic_state_export.py --domain "$DOMAIN" --batches 1 64 256 1024 --reference-classic tests/helpers/reference/classic_column_reference.py --output "$BASE/results/$DOMAIN.json"
