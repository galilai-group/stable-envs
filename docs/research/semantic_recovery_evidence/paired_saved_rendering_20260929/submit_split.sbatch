#!/bin/bash
#SBATCH --job-name=se-paired-submit
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/submit-split-%j.log
set -euo pipefail
sbatch --parsable --partition=3090-gcondo --qos=cs-3090-gcondo --array=0-31%32 --mem=32G --time=00:45:00 /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/gpu.sbatch > /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/condo-id.txt
sbatch --parsable --array=32-41%2 --mem=32G --time=00:45:00 /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/gpu.sbatch > /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/norm-id.txt
