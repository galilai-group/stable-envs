#!/bin/bash
#SBATCH --job-name=se7-craftax-full-gpu-ref
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# R for the GPU tier: the official renderer and ours, on the SAME CARD in the SAME JOB.
#
# Two interpreters in one allocation, because no single venv can do it and neither may be modified:
#   $SCR/venvs/se7-craftax-gpu   W1's, jax[cuda12] 0.11.1 + craftax -- but NO TORCH, so it cannot
#                                run our renderer. Reused, not rebuilt, and not added to.
#   $SCR/venvs/se7-craftax       ours, torch + craftax -- but jax[cpu], so jax.default_backend() is
#                                'cpu' in here whatever JAX_PLATFORMS says. Job 6409535 would have
#                                timed the "GPU reference" on the CPU and was cancelled before it ran.
# gpu_official.py asserts jax.default_backend() == 'gpu' and exits non-zero rather than produce a
# number that never touched the card.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
unset JAX_PLATFORMS
GVENV=$SCR/venvs/se7-craftax-gpu
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
R=examples/results/step7/craftax-full/sweep
H=$(hostname)
mkdir -p $R
echo "== host $H =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== the OFFICIAL renderer, on the card (W1's cuda venv) =="
"$GVENV/bin/python" $P/gpu_official.py --size 10 --reps 300 --tag "$H" \
  --out $R/gpu_official_$H.json

echo "== OUR renderer, same card, same job (our venv) =="
"$VENV/bin/python" $P/bench.py --device cuda --tag "ref_$H" --batches 256,4096 --sizes 10 --seconds 2.0
