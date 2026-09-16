#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-owner2-ng
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Which test BUILT the renderer that the four failing tests are charged with closing?
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6/owner2_normgpu
export PYTHONPATH=$WS:$SCR/se7/swm-src:$WS/scratch/step7/ogb-manipulation
mkdir -p "$OUT"
cd "$WS"
WANT=981ab3c9d41bc700b273e85e7ae690ff
GOT=$(md5sum stable_envs/envs/ogb_manip.py | cut -d' ' -f1)
echo "ogb_manip.py md5 want=$WANT got=$GOT"
[ "$WANT" = "$GOT" ] || { echo "MD5 MISMATCH"; exit 3; }
set -x
"$VENV/bin/python" -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -k "OGBCube or OGBScene" -q --no-header -rf --tb=line \
  -p owner_plugin > "$OUT/owner_in_pytest.txt" 2>&1 || true
grep -E "RENDERER (BORN|CLOSED|FINALISED|SUMMARY)|^FAILED|passed|failed|resources:" \
  "$OUT/owner_in_pytest.txt" | head -60
cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
