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
# SLURM copies the script to its spool dir, so $0 is NOT in the worktree: source the
# header from the workspace by absolute path.
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
# JAX_PLATFORMS stays "cpu": the reference is a correctness oracle, not a speed measurement, and
# it is called UN-JITTED, so putting it on the GPU buys nothing and would have jax and torch
# contending for the same device while the torch timings are being taken. Only torch uses the
# card here, which is the point of the job.
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py
"$VENV/bin/python" scratch/step7/craftax-classic-full/certify.py --device cuda --tag "$(hostname)"
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cuda \
  --tag "$(hostname)"
PYTHONPATH=$SCR/se6/baseline "$VENV/bin/python" examples/bench_step3.py --device cuda \
  --envs CraftaxClassic || echo "baseline leg unavailable"
