#!/bin/bash
# Push this worktree to Oscar with the HEAD SHA stamped into GIT_COMMIT (rsync carries no .git, and
# examples/fidelity_step4.py + the sweep wrapper read STABLE_ENVS_GIT_COMMIT / GIT_COMMIT for provenance).
set -e
WT=/Users/samib/stable-envs-step6-cpu-3d
git -C "$WT" rev-parse HEAD > "$WT/GIT_COMMIT"
[ -z "$(git -C "$WT" status --porcelain)" ] || echo "WARNING: dirty tree; GIT_COMMIT names HEAD, not the tree" >&2
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' \
  --exclude 'examples/results/step6/cpu-3d/fetch_feasibility/*.bin' \
  "$WT/" sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se6/cpu-3d/
echo "synced $(cat "$WT/GIT_COMMIT")"
