#!/bin/bash
#SBATCH --job-name=se6-raster2d-before
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6 raster2d PROFILE-BEFORE, on Oscar's batch partition at the certification thread budget.
#
# "Before" here means: the baseline renderer, with the device-dispatch defect fixed and NOTHING
# else. Without that fix PushT and SimplePointMaze raise on any Triton host and there is no CPU
# number to be before anything. Every other line of the renderer is the frozen baseline.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=raster2d
# TREE is the code being measured; the artifacts always land under the workstream tree, so a before
# taken from the pre-patch copy and an after taken from the patched one end up side by side.
PHASE=${PHASE:-before}
TREE=${TREE:-$SCR/se6/$WS}
export PYTHONPATH=$TREE
PY=$SCR/venvs/se-step6/bin/python
cd $TREE
OUT=$SCR/se6/$WS/examples/results/step6/$WS/profile_$PHASE
STATES=$TREE/examples/results/step4/states
set -x
$PY -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

FAILED=0
# SimplePointMaze is profiled to batch 64 and not 256. At the baseline one 500 px frame is 325 ms
# and a batch of 64 is 76 SECONDS per call, so 13 calls at 256 is over four hours for one cell of
# one note -- a number that would say nothing the 64 already says, bought by losing the other five
# envs to the wall clock. The certification sweep, which does have to reach the frozen batches, caps
# each cell with --max-cell-s instead. The AFTER run uses this same list, because a before and an
# after taken at different batch sizes are not a before and an after.
ENVS=${ENVS:-"PushT SimplePointMaze TwoRoom TwoRoomTarget Piecewise PiecewiseZones"}
for E in $ENVS; do
  case $E in SimplePointMaze) BATCHES="1 16 64";; *) BATCHES="1 16 64 256";; esac
  $PY scratch/step6/profile_note.py --env "$E" --device cpu --threads 1 \
      --batch $BATCHES --repeat 10 --warmup 3 --workstream "step6/$WS" \
      --states-dir "$STATES" --out "$OUT" || { echo "PROFILE $E rc=$?"; FAILED=$((FAILED+1)); }
  $PY scratch/step6/raster2d/attribute.py --env "$E" --device cpu --threads 1 \
      --batch 1 64 --repeat 8 --warmup 2 --states-dir "$STATES" --out "$OUT" \
      || { echo "ATTRIB $E rc=$?"; FAILED=$((FAILED+1)); }
done

# CPU pixel identity for the six configurations, on the certification machine.
$PY scratch/step6/raster2d/cpu_digest.py --threads 1 --states-dir "$STATES" \
    --out "$SCR/se6/$WS/examples/results/step6/$WS/cpu_digest_$PHASE.json" \
    || { echo "DIGEST rc=$?"; FAILED=$((FAILED+1)); }

echo "RASTER2D_PROFILE_$PHASE failed=$FAILED"
ls "$OUT"/*.json | wc -l
test "$FAILED" -eq 0
