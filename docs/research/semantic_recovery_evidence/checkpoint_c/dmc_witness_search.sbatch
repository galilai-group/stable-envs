#!/bin/bash
#SBATCH --job-name=se-dmc-witness-search
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/witness-search-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2
export PYTHONPATH="$base/dmc-aligned-diagnostic-c6ae835:$base/swm-reference:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
cd "$base/dmc-aligned-diagnostic-c6ae835"
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python "$base/dmc_witness_search-c6ae835.py"
