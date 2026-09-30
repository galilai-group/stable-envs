#!/bin/bash
#SBATCH --job-name=se-scene-kinematics-stage
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/stage-%j.log
set -euo pipefail
cp -a /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/source-cpu /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/source
cp -a /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/overlay/. /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/source/
sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/gpu.sbatch > /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/gpu-job-id.txt
sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/supplemental.sbatch > /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/supplemental-job-id.txt
sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/cpu.sbatch > /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/cpu-job-id.txt
