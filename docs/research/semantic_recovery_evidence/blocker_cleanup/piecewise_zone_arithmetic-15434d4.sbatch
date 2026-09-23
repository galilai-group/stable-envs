#!/bin/bash
#SBATCH --job-name=se-piecewise-diag
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/piecewise-arithmetic-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
export PYTHONPATH=/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python "$base/${1:?immutable diagnostic filename}" > "$base/${2:?result filename}"
echo PIECEWISE_ZONE_ARITHMETIC_COMPLETE
