#!/bin/bash
#SBATCH --job-name=se-dmc-components-stage
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-cpu-components-gate-20260929/stage-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/dmc-cpu-components-gate-20260929
cp -a /oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928/source-v12 "$root/source"
cp -a "$root/overlay/." "$root/source/"
