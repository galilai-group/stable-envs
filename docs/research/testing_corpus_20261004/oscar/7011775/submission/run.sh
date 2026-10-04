#!/bin/bash
#SBATCH --job-name=se-testing-corpus
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=gpu3001
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/testing-corpus-roundoff-cc81789c-v1/job-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/testing-corpus-roundoff-cc81789c-v1
RUN="$BASE/results/$SLURM_JOB_ID"
mkdir -p "$RUN/code" "$RUN/cache"
tar -xzf "$BASE/code.tar.gz" -C "$RUN/code"
cd "$RUN/code"
export PYTHONPATH="$PWD:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$RUN/cache" CUDA_CACHE_PATH="$RUN/cache/cuda" MPLCONFIGDIR="$RUN/cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
nvidia-smi > "$RUN/nvidia-smi.txt"
lscpu > "$RUN/lscpu.txt"
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python "$BASE/validate.py" "$BASE" "$RUN"
