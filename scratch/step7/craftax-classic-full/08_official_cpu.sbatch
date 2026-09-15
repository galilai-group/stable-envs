#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-official-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The official reference, timed the way Sami's ruling defines it, ON ONE CORE.
# "One core" is enforced by the CGROUP: --cpus-per-task=1, so BOTH our renderer and the reference
# get exactly one CPU whatever either asks for. That is the point -- XLA's CPU backend runs its own
# thread pool and ignores OMP_NUM_THREADS, which is why the earlier 610.6 f/s could not be used as a
# one-core reference. An earlier attempt used `taskset -c 0` and failed with "Invalid argument"
# because SLURM's cpuset does not necessarily contain cpu 0; the cgroup already does this job.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="; cat /proc/loadavg; nproc
echo "allowed cpus: $(python3 -c 'import os;print(sorted(os.sched_getaffinity(0)))')"
"$VENV/bin/python" scratch/step7/craftax-classic-full/official_timing.py \
  --device cpu --threads 1 --batch 64 --tag "$(hostname)"
cat /proc/loadavg
