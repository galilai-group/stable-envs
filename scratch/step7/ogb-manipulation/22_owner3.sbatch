#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-owner3
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Two questions, one job.
# (A) ORDERING: do the four rows fail when run ALONE? If they pass alone, the renderer they are
#     charged with closing was built by an earlier test and this is an attribution artefact.
# (B) OWNERSHIP: with __new__ patched as well as __init__, and with capture OFF (-s) so no birth
#     line is swallowed by pytest, which call builds the four that die.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6/owner3
export PYTHONPATH=$WS:$SCR/se7/swm-src:$WS/scratch/step7/ogb-manipulation
mkdir -p "$OUT"
cd "$WS"
set -x
echo "=== (A) the four rows ALONE ==="
"$VENV/bin/python" -m pytest tests/gpu/test_reproducibility.py \
  -k "(OGBCube or OGBScene) and (a_shared_style_broadcasts or dataset_paths_regenerate)" \
  -q --no-header -rf --tb=line > "$OUT/alone.txt" 2>&1 || true
grep -E "^(FAILED|ERROR)|passed|failed|resources:" "$OUT/alone.txt" | tail -5

echo "=== (B) full OGB selection, capture OFF, __new__ patched ==="
"$VENV/bin/python" -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -k "OGBCube or OGBScene" -q --no-header -rf --tb=line -s \
  -p owner_plugin > "$OUT/owner3.txt" 2>&1 || true
grep -E "RENDERER (NEW|CLOSED|FINALISED)|^FAILED|passed|failed|resources:" "$OUT/owner3.txt" | head -70
cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
