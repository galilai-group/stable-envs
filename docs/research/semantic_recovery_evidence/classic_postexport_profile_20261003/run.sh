#!/bin/bash
#SBATCH --job-name=classic-profile
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-postexport-profile-v1-20261003/profile-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-postexport-profile-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
nvidia-smi
lscpu
FAMILIES=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
"$PY" scripts/benchmarks/online_stepping/classic_profile.py --family "$FAMILY" --batches 1 64 256 1024 --devices cpu cuda --repeats 6 --output "$BASE/results/$FAMILY.json"
