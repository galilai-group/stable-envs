#!/bin/bash
#SBATCH --job-name=se-fetch-obs-gate
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/fetch-observation-acceptance-20261001-v1/terminal.log
set -euo pipefail
cd /oscar/scratch/sboughan/se-semantic-recovery/fetch-observation-acceptance-20261001-v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH=/oscar/scratch/sboughan/se-semantic-recovery/fetch-observation-acceptance-20261001-v1
export XDG_CACHE_HOME=/oscar/scratch/sboughan/se-semantic-recovery/fetch-observation-acceptance-20261001-v1/cache
nvidia-smi
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_probe.py --baseline-fetch baseline_fetch.py --output cpu.json --device cpu --gate-only --gate-batch 64
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_probe.py --baseline-fetch baseline_fetch.py --output cuda.json --device cuda --gate-only --gate-batch 64
