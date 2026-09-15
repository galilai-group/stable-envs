#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# GPU tier and the CUDA leg of Gate A, before and after in ONE allocation on ONE card.
# One GPU job per agent at a time (norm-gpu allows 2 across all agents); never cancel another
# agent's job, and never a simclr_cifar10 job.
set -euo pipefail
source "$(dirname "$0")/env_header.sh"
unset JAX_PLATFORMS
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py
"$VENV/bin/python" scratch/step7/craftax-classic-full/certify.py --device cuda --tag "$(hostname)"
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cuda \
  --tag "$(hostname)"
PYTHONPATH=$SCR/se6/baseline "$VENV/bin/python" examples/bench_step3.py --device cuda \
  --envs CraftaxClassic || echo "baseline leg unavailable"
