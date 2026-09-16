#!/bin/bash
# Merge-gate item 5 for Workstream B: the Step-4 frozen Fetch corpora through both trees, one card.
#
# Run inside the held allocation rather than as its own job -- the whole thing is four 64x480 renders
# per tree and the allocation is already warm:
#
#   srun --jobid=<alloc> --overlap -n 1 --cpus-per-task=6 --gres=gpu:1 \
#       bash $SCR/se8/variations-fetch/scratch/step8/variations-fetch/frozen_replay.sh
#
# If the allocation is gone, the same body runs as a <=30 min sbatch on 3090-gcondo with
# --constraint=geforce3090 --exclude=gpu2108,gpu2113,gpu2110,gpu2116.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1

TREE=${TREE:-$SCR/se8/variations-fetch}
BASE=${BASE:-$SCR/se8/baseline}
FETCHPY=$SCR/venvs/se-s3b-fetch311/bin/python

# .git is not synced to the cluster (the trees are rsync copies), so the SHA of each tree is stamped
# into a .git_sha file by the sync and read back here rather than derived on the node.
export STABLE_ENVS_GIT_COMMIT=$(cat "$TREE/.git_sha" 2>/dev/null || echo unknown)
export SE8_BASELINE_SHA=$(cat "$BASE/.git_sha" 2>/dev/null || echo unknown)
export SE8_VENV='$SCR/venvs/se-s3b-fetch311'
export BASE_TREE=$BASE

echo "=============================================================================="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) date=$(date -u +%FT%TZ)"
echo "tree=$TREE sha=$STABLE_ENVS_GIT_COMMIT"
echo "base=$BASE sha=$SE8_BASELINE_SHA"
echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no gpu"
echo "=============================================================================="

cd "$TREE" || exit 1
PYTHONPATH=$TREE $FETCHPY scratch/step8/variations-fetch/frozen_corpus_replay.py
echo "REPLAY rc=$?"
