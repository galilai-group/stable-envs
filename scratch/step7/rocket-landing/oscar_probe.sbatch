#!/bin/bash
#SBATCH --job-name=se7-rocket-probe
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# M4 diagnostic only: probe_batch_size.py on the cf355a3 tree (no measurement, no declaration).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
module load cuda/12.9.0 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
PY=$SCR/venvs/se7-rocket/bin/python
WS=$SCR/se7/rocket-landing
[ -f "$WS/.git_sha" ] || { echo "MISSING $WS/.git_sha"; exit 3; }
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) SHA $(cat "$WS/.git_sha") LOAD $(cat /proc/loadavg | cut -d" " -f1-3)"
cd "$WS"
for P in probe_batch_size.py probe_clip_ulps.py; do PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/$P" 2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr"; done
echo "S7PROBE rc=$?"
