#!/bin/bash
# Push this worktree to Oscar and stamp the SHA it is at, then submit.
#
#   scratch/step6/raster2d/sync.sh [sbatch script] [extra sbatch args...]
#
# Two things it exists to stop, both of which have already happened here:
#   * `rsync --delete` mirrors the worktree and DELETES the results a running job just wrote into
#     examples/results/step6/raster2d on the node -- so that path is excluded, always;
#   * a shard naming a commit that does not contain the code it timed. GIT_COMMIT is written from
#     `git rev-parse HEAD` at submit time, and a dirty worktree is refused rather than stamped with
#     the SHA of something else.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
REMOTE=sboughan@sshcampus.ccv.brown.edu
TREE=/oscar/scratch/sboughan/se6/raster2d
cd "$ROOT"
if [ -n "$(git status --porcelain -- stable_envs scratch tests examples/bench_step3.py)" ]; then
    echo "refusing to sync: commit or stash the code first, or the SHA stamp will be a lie" >&2
    git status --short -- stable_envs scratch tests examples/bench_step3.py >&2
    exit 1
fi
rsync -az --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' \
      --exclude 'examples/results/step6/raster2d' "$ROOT/" "$REMOTE:$TREE/"
git rev-parse HEAD | ssh -o BatchMode=yes "$REMOTE" "cat > $TREE/GIT_COMMIT"
echo "synced $(git rev-parse --short HEAD) to $TREE"
[ $# -eq 0 ] && exit 0
ssh -o BatchMode=yes "$REMOTE" "cd $TREE && sbatch ${*}"
