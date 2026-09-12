#!/bin/bash
#SBATCH --job-name=se-step3-gpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# --constraint=geforce3090 alone, not the usual geforce3090|l40s: Step 2 named the RTX 3090 as the
# certification device and an L40S has 5x the memory bandwidth of one, so a sweep that silently landed
# on an L40S would be a different measurement wearing the 3090's label.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch
export MUJOCO_GL=egl MPLBACKEND=Agg
# warp resolves its kernel cache from its OWN path, not XDG_CACHE_HOME, and lands in HOME --
# which is already SOFT_EXCEEDED at 100.88/100 GB with a 10-day grace. Measured at 2.8 MB so it
# is not what fills HOME, but nothing in this campaign has a reason to write there.
export WARP_CACHE_PATH=$SCR/.cache/warp
export PYTHONPATH=$SCR/stable-envs
VENV=$SCR/venvs/se-step3

cd $SCR/stable-envs || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK load=$(cat /proc/loadavg)"

"$VENV/bin/python" examples/bench_step3.py --device cuda \
    --out "$SCR/step3-out/gpu/${OUTNAME:-3090}.json" \
    --envs "${ENVS:-auto}" --resume --max-cell-s "${MAXCELL:-30}" ${EXTRA:-}
echo "GPU SWEEP DONE rc=$?"
