#!/bin/bash
#SBATCH --job-name=se6-raster2d-swmnav-exp
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; AFT=$SCR/se6/$WS; RES=$AFT/examples/results/step6/$WS/swmnav2
mkdir -p "$RES"; cd $AFT
echo "node=$(hostname) job=$SLURM_JOB_ID"
set -x
PYTHONPATH=$AFT $SCR/venvs/se-step5/bin/python scratch/step6/raster2d/swmnav_exp_probe.py \
    --n 4194304 --out "$RES/exp_spelling.json"
echo "SWMNAV_EXP_DONE rc=$?"
