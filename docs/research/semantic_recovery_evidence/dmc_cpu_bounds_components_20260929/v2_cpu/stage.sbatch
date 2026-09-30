#!/bin/bash
#SBATCH --job-name=se-dmc-components-v2-stage
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-components-v2-gate-20260929/stage-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/dmc-components-v2-gate-20260929
cp -a /oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-gate-20260929/source "$root/source"
cp -a "$root/overlay/." "$root/source/"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/source:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/source"
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python -m pytest "$root/tests/test_raycast3d_cpu_components.py" tests/unit/test_raycast3d_primitive_bounds.py -q --junitxml="$root/pytest.xml"
