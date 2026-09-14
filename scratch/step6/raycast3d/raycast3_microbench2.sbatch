#!/bin/bash
#SBATCH --job-name=se6-raycast3d-mb2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
rm -rf $TORCH_EXTENSIONS_DIR/raycast3d_microbench*
$PY scratch/step6/raycast3d/cast_microbench.py --threads 1 --rays 7056 --repeat 9 --warmup 3 \
    --out "$OUT/raycast3_cast_microbench.json" || echo "MB rc=$?"
MB_NO_MARCH=1 $PY scratch/step6/raycast3d/cast_microbench.py --threads 1 --rays 7056 --repeat 9 \
    --warmup 3 --out "$OUT/raycast3_cast_microbench_nomarch.json" || echo "MBNM rc=$?"
echo "MB2 done"
