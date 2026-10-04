#!/bin/bash
#SBATCH --job-name=ogbench-errors-submit
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-identity-v3-20261003/submit-%j.log
set -euo pipefail
cd /oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-identity-v3-20261003
mkdir submission-started
sha256sum -c bundle.sha256
tar -xzf bundle.tar.gz
sbatch --parsable run.sh | tee array-job.txt
sbatch --parsable cpu.sh | tee cpu-array-job.txt
