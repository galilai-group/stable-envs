#!/bin/bash
# Run ONE stage of the closure sweep inside a HELD SLURM allocation, from the login node.
#
#     bash scratch/step7/closure/run_in_alloc.sh <alloc-jobid> <stage-list> <tag> [extra env...]
#
# Sami's rule (2026-09-16): nothing that computes may run on an Oscar login node. `srun` itself is a
# client that forwards I/O and does no work, so the login node only ever holds that client; every
# line of Python, every render and every pytest runs on the allocation's compute node.
#
# `nohup ... &` so an ssh hiccup cannot kill the stage: srun dies with its client, and losing an hour
# of replay to a dropped connection is the failure this avoids. Output goes to $SCR/logs, never into
# the repository -- it carries absolute interpreter paths that the citation scanner would fail on.
set -uo pipefail
ALLOC=$1; STAGES=$2; TAG=$3; shift 3
export SCR=/oscar/scratch/$USER
TREE=$SCR/se7/closure
LOG=$SCR/logs/closure2_${TAG}.log
cd "$TREE" || exit 1
nohup srun --jobid="$ALLOC" --overlap -n 1 --cpus-per-task=6 \
      bash -lc "cd $TREE && OUTROOT=$TREE/examples/results/step7/closure2 STAGES='$STAGES' $* bash scratch/step7/closure/closure_gpu.sbatch" \
      > "$LOG" 2>&1 &
echo "STARTED stage '$STAGES' tag=$TAG alloc=$ALLOC pid=$! log=$LOG"
