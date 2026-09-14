#!/bin/bash
#SBATCH --job-name=se6-raster2d-gpuab
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Does swm_nav's cuda branch cost anything? Asked in one process, interleaved, on one warmed card --
# see the script's docstring for why two SLURM legs cannot answer it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; AFT=$SCR/se6/$WS
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID tree $AFT at $(cat $AFT/GIT_COMMIT 2>/dev/null || echo unknown)"
set -x
cd $AFT
PYTHONPATH=$AFT $SCR/venvs/se-step5/bin/python scratch/step6/raster2d/gpu_same_process_ab.py \
    --device cuda --batch 1 16 64 256 1024 4096 \
    --out "$AFT/examples/results/step6/$WS/sweep_r4/gpu_same_process_ab.json"
echo "GPU_AB_DONE"
