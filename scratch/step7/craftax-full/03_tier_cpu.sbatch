#!/bin/bash
#SBATCH --job-name=se7-craftax-full-tier-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The single mixed-corpus tier, the five band diagnostics and both official denominators, on a batch
# node at one torch thread. --exclusive and /proc/loadavg on both sides because W1 measured a
# contended node at 6,219 f/s where the same code does 126,845 on a laptop, and a tier taken on a
# contended node is not a tier.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
echo "== host $(hostname) arch $(uname -m) =="; cat /proc/loadavg
"$VENV/bin/python" $P/bench.py --device cpu --tag "$(hostname)" --official --seconds 2.0
cat /proc/loadavg
