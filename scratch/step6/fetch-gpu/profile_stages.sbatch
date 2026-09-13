#!/bin/bash
#SBATCH --job-name=se6-fetch-gpu-stages
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6D profile-before, part 2: WHY the Fetch MJWarp path does not amortise.
#
# The existing note (profile_before/*_gpu2104.*) stopped at B=1024 and attributes at OPERATOR
# granularity. This adds (a) B=4096, so the frozen BATCH_SCALES and NO_LARGE_BATCH_REGRESSION
# qualifiers are measured rather than extrapolated, (b) an exhaustive wall partition of render()
# by the functions the env already calls, (c) a device partition with memcpy BYTES, and (d) the
# controlled experiment: the same code path at 84/120/240/480 px.
#
# geforce3090 by name so the numbers sit on the same card class as Step 3 (gpu2101/2107) and the
# existing profile-before (gpu2104). An L40S would make this a comparison between two machines.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip HF_HOME=$SCR/.cache/hf
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export WS=$SCR/se6/fetch-gpu
export PYTHONPATH=$WS
cd "$WS"
PY=$SCR/venvs/se-s3b-fetch311/bin/python
set -x
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import mujoco, mujoco_warp, warp, torch; print('mujoco', mujoco.__version__, 'mjw', mujoco_warp.__version__, 'warp', warp.__version__, 'torch', torch.__version__)"

OUT=$WS/examples/results/step6/fetch-gpu

# (a) the frozen harness, extended to B=4096, so profile_before covers the qualifier batch.
for E in FetchReach FetchPickAndPlace; do
  $PY scratch/step6/profile_note.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --workstream step6/fetch-gpu \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_before_4096" || echo "profile_note $E rc=$?"
done

# (b,c,d) stage-resolved partition + the resolution experiment.
$PY scratch/step6/fetch-gpu/profile_stages.py --env FetchReach \
    --batch 1 16 64 256 1024 4096 --sizes 84 120 240 480 --size-batch 1 256 4096 \
    --repeat 10 --warmup 3 --out "$OUT/profile_stages" || echo "stages Reach rc=$?"

$PY scratch/step6/fetch-gpu/profile_stages.py --env FetchPickAndPlace \
    --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
    --out "$OUT/profile_stages" || echo "stages PnP rc=$?"

echo "STEP6D_DONE"
ls -la "$OUT"/profile_stages "$OUT"/profile_before_4096 2>/dev/null
