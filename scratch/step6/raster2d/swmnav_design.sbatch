#!/bin/bash
#SBATCH --job-name=se6-raster2d-swmnav-design
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# SWMNAV-2, design phase. Two questions, one job, before a line of kernel is written:
#   1. where does the cuda time go per stage, and how many launches does a render issue;
#   2. can a Triton kernel reproduce this module's arithmetic BIT-FOR-BIT -- in particular the
#      float64 exp, which no kernel in the repo has had to match before.
set -uo pipefail
# The SHA is stamped by the submitting worktree (see scratch/step6/raster2d/sync.sh), not read from
# whatever the tree last carried: gpu_after.json, gpu_profile_after.json and replay/swmnav2.json
# were measured before their commit existed and name its parent. swmnav2/PROVENANCE_NOTE.md holds
# the md5s that say what they actually measured.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; AFT=$SCR/se6/$WS; RES=$AFT/examples/results/step6/$WS/swmnav2
mkdir -p "$RES"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID tree $AFT at $(cat $AFT/GIT_COMMIT 2>/dev/null || echo unknown)"
set -x
cd $AFT
PY=$SCR/venvs/se-step5/bin/python

PYTHONPATH=$AFT $PY scratch/step6/raster2d/swmnav_gpu_profile.py --device cuda \
    --batch 256 1024 4096 --out "$RES/gpu_profile_before.json"

# with fusion ON first, so the table shows what the knob is worth rather than asserting it
PYTHONPATH=$AFT $PY scratch/step6/raster2d/swmnav_exact_probe.py --n 4194304 --fusion \
    --out "$RES/exact_probe_fusion_on.json" || true
PYTHONPATH=$AFT $PY scratch/step6/raster2d/swmnav_exact_probe.py --n 4194304 \
    --out "$RES/exact_probe.json"
echo "SWMNAV_DESIGN_DONE rc=$?"
