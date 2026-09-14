#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r4units
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The CPU-side invariance the ticket asks for. tests/gpu/test_reproducibility.py owns purity and
# batch/chunk invariance but is cuda-only and skips the module without a device, so it has nothing to
# say about a switch that only changes the CPU cast. The pure-torch case below is the one that
# matters for THIS design: the bound's ray set is a union over the batch, so batching changes which
# rays reach the intersector, and "row 0 alone == row 0 in company" is a claim rather than an
# identity.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_raycast3d_primitive_bounds.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -8
echo "R4_UNITS done"
