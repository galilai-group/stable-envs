#!/bin/bash
#SBATCH --job-name=se-step4
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# The RTX 3090 by name, as in Step 3: an L40S would satisfy the usual geforce3090|l40s constraint and
# would be a different machine wearing the certification device's label. Gate A does not depend on
# which GPU drew the frame, but the frame DIGESTS this step freezes do, and Step 6 replays them.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg
export PYTHONPATH=$SCR/stable-envs
VENV=${VENV:-$SCR/venvs/se-step3}
SHARD=${SHARD:?set SHARD=dmc|ogb|raster|fetch|swm}

# One job may cover several shards, and on this cluster it usually should: a shard is minutes of
# work (OGBench measures at ~40s per configuration) while a GPU slot is hours of queue, so asking for
# three allocations to do twenty minutes of rendering is the expensive way round. SHARD=rest runs the
# three that share the se-step3 interpreter, one after another, in one allocation.
case "$SHARD" in
  rest) LIST="ogb raster" ;;
  *)    LIST="$SHARD" ;;
esac

# Which configurations get a human comparison sheet, per shard. Here rather than in --export because
# sbatch splits --export on commas, so a multi-value variable passed that way silently arrives as one
# name plus several bogus exports. Section 11 asks for a representative set and not for all 49: a
# sheet per configuration is ~40 MB of PNG in a repository whose thesis is that frames regenerate.
sheets_for() {
  case "$1" in
    dmc)    echo "DMCReacherHard,DMCBallInCupCatch,DMCWalkerWalk,DMCHumanoidStand" ;;
    ogb)    echo "PointmazeLargeNavigate" ;;
    raster) echo "PushT" ;;
    fetch)  echo "FetchPickAndPlace" ;;
    swm)    echo "CraftaxClassic,SimplePointMaze" ;;
    *)      echo "" ;;
  esac
}
cd $SCR/stable-envs || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID shard=$SHARD venv=$VENV load=$(cat /proc/loadavg)"

rc=0
for S in $LIST; do
  echo "=== shard $S ==="
  "$VENV/bin/python" examples/fidelity_step4.py \
      --shard "$S" --device cuda --resume \
      --states-dir "$SCR/step4-out/states" \
      --images "$SCR/step4-out/comparisons" \
      --image-configs "$(sheets_for $S)" \
      --out "$SCR/step4-out/conformance/${S}.json"
  echo "SHARD $S DONE rc=$?"
done
exit $rc
