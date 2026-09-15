#!/bin/bash
#SBATCH --job-name=se7-craftax-full-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# PHASE 1 ONLY -- do not submit before the renderer exists. The CUDA leg of Gate A and the GPU
# tier, before and after in ONE allocation on ONE card. One GPU job in flight for this workstream;
# never cancel a job this workstream did not submit, and never a simclr job.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
unset JAX_PLATFORMS
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
echo "== host $(hostname) =="; nvidia-smi --query-gpu=name,memory.total --format=csv
"$VENV/bin/python" $P/certify.py --device cuda --tag "$(hostname)"
