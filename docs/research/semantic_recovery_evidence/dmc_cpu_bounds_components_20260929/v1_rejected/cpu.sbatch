#!/bin/bash
#SBATCH --job-name=se-dmc-components-gate
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --array=44,64,67%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-cpu-components-gate-20260929/cpu-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
root=/oscar/scratch/sboughan/se-semantic-recovery/dmc-cpu-components-gate-20260929
for leg in baseline candidate; do
  /usr/bin/python3 "$root/$leg/profile_oscar.py" "$root/$leg" "$SLURM_ARRAY_TASK_ID"
done
