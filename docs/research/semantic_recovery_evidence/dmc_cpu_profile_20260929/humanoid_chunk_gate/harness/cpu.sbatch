#!/bin/bash
#SBATCH --job-name=se-humanoid-chunk-gate
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-gate-20260929/cpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
root=/oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-gate-20260929
cp -a /oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928/source-v12 "$root/source"
cp -a "$root/overlay/." "$root/source/"
for leg in baseline candidate; do
  /usr/bin/python3 "$root/$leg/profile_oscar.py" "$root/$leg" 66
done
