#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6rpt
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nodelist=node1946
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# THE ONE authorized repeat of the RAYCAST-6 certification shard, for the two Pointmaze rows only.
#
# Why: PointmazeMediumNavigate's crossing is cores_cpu 1.207 on its own leg's decode rate and 1.020
# on the decode rate measured for the SAME configuration in the SAME job's other leg -- the one-core
# JPEG bar moved 2,678 -> 2,264 f/s (-15%) between legs on node2317, which is a property of the node
# rather than of the renderer. A second reading on a quieter node is the only thing that separates
# "it crosses" from "it crossed on that node that afternoon".
#
# Two guards, both of which REJECT AND RECORD rather than publish:
#   --max-load 12          a pre-condition on the measurement, checked in a second before anything
#                          is timed, rather than discovered afterwards in the reference spread. The
#                          48-core nodes in the 23xx block -- where every earlier job in this ticket
#                          landed -- were sitting at load 31-48 when this was written, while node1946
#                          was at 4.01 with 16 idle cores, which is why it is named above. An
#                          --exclude of a node RANGE is refused by this cluster ("Invalid node name
#                          specified") because the range spans names that do not exist, so the quiet
#                          node is requested rather than the busy ones excluded.
#   --max-reference-spread 1.25   the same bound the first pair cleared at 1.09/1.07.
# A rejected shard is still written, and marked. A missing file looks like a job that never ran.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$SCR/tmp; mkdir -p "$TMPDIR"
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
SW=$WS/examples/results/step6/raycast3d/sweep/r6_repeat
mkdir -p "$SW"
# rsync drops .git, so the SHA under measurement reaches the shard through the environment. The
# sync writes $WS/GIT_COMMIT; without it certify_shard_cpu3d.py records tree_commit "unknown", which
# is what the first pair did and what this run exists partly to stop.
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/GIT_COMMIT" 2>/dev/null || echo unknown)
set -x
cd "$WS"
hostname; lscpu | grep -E "^Model name" | head -1; cat /proc/loadavg
echo "STABLE_ENVS_GIT_COMMIT=$STABLE_ENVS_GIT_COMMIT"

ENVS="PointmazeLargeNavigate,PointmazeMediumNavigate"
SE6_LEG=before SE6_TREE_COMMIT=${INT_COMMIT:-5b327af} PYTHONPATH=$INT $PY scratch/step6/raycast3d/r6_certify_shard.py \
    --device cpu --threads 1 --envs "$ENVS" --reps 25 --max-cell-s 20 \
    --max-reference-spread 1.25 --max-load 12 \
    --out "$SW/cpu_1thread_before_integration.json" || echo "RPT_BEFORE rc=$?"
SE6_LEG=after SE6_TREE_COMMIT=$STABLE_ENVS_GIT_COMMIT PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_certify_shard.py \
    --device cpu --threads 1 --envs "$ENVS" --reps 25 --max-cell-s 20 \
    --max-reference-spread 1.25 --max-load 12 \
    --out "$SW/cpu_1thread_after_branch.json" || echo "RPT_AFTER rc=$?"

echo "######## the two readings, side by side"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_tiers.py --sweep sweep/r6 2>&1 | tail -8
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_tiers.py --sweep sweep/r6_repeat 2>&1 | tail -8
echo "R6_REPEAT done"
