#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-sweep-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The CPU tier, measured per lighting band and then on the mixed corpus.
#
# Reporting one number here would be dishonest in both directions: the daylight path is untouched
# by Step 7 and keeps the Step-6 throughput, while a night frame carries a per-pixel float
# post-process that the reference performs and that this renderer must reproduce bit for bit. The
# certified number is the mixed-corpus one; the band numbers say where it comes from.
set -euo pipefail
# SLURM copies the script to its spool dir, so $0 is NOT in the worktree: source the
# header from the workspace by absolute path.
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cpu --threads 1 \
  --tag "$(hostname)"
# and the baseline, so before/after are the same node in the same job
PYTHONPATH=$SCR/se6/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic || echo "baseline leg unavailable"
