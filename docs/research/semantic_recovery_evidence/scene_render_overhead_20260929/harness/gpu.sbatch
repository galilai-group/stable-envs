#!/bin/bash
#SBATCH --job-name=se-scene-render-overhead
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/gpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cp -a /oscar/scratch/sboughan/se-semantic-recovery/cube-context-budget-20260929/source /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/source
cp -a /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/overlay/. /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/source/
for variant in baseline candidate; do
 /usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/$variant/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/$variant 30
done
