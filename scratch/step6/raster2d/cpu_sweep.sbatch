#!/bin/bash
#SBATCH --job-name=se6-raster2d-cpusweep
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6 raster2d: the 1-thread CPU certification sweep, BEFORE and AFTER, in ONE job on ONE node.
#
# Two shards, one per interpreter, exactly as Step 3 did it, because no interpreter on Oscar holds
# every official renderer:
#   se-step3  -> gym_pusht + pymunk  -> PushT is the only row with a reference there
#   se-step5  -> stable_worldmodel + matplotlib -> SimplePointMaze, TwoRoom, TwoRoomTarget,
#                Piecewise, PiecewiseZones
# Both venvs carry torch 2.14.0+cu126 and triton 3.8.0, so both hit the device-dispatch defect that
# made these two rows CPU_UNMEASURED in Step 3; the BEFORE tree is therefore the baseline WITH that
# fix and nothing else, so the sweep measures the optimisation rather than the defect.
#
# --states-out on the before run and --states-in on the after run: the two halves of a before/after
# have to render the same states or the ratio is between two different workloads.
#
# GIT_COMMIT is written into each tree by the sync from the Mac, not committed:
#   git rev-parse HEAD > /tmp/GIT_COMMIT && scp /tmp/GIT_COMMIT <tree>/GIT_COMMIT
# A committed one would be a SHA that is always one commit stale.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=raster2d
BEF=$SCR/se6/$WS-preB          # baseline + the device-dispatch fix only
AFT=$SCR/se6/$WS               # + the bounded/dispatched primitive loop
TAG=${TAG:-}
OUT=$SCR/se6/$WS/examples/results/step6/$WS/sweep$TAG
ST=$OUT/states
mkdir -p "$OUT" "$ST"
# rsync drops .git, so the SHA reaches the shard through the environment or a
# GIT_COMMIT file at the tree root -- otherwise every artifact records git_commit: null.
echo "before tree $BEF at $(cat "$BEF/GIT_COMMIT" 2>/dev/null || echo unknown)"
echo "after  tree $AFT at $(cat "$AFT/GIT_COMMIT" 2>/dev/null || echo unknown)"
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK load=$(cat /proc/loadavg)"
lscpu | grep "Model name"
set -x

run () {   # tree, tag, venv, envs, states-flag
  # PER LEG, not once for the job: exporting one SHA globally stamped the before shard with the
  # after tree's commit. stable_envs_file still discriminated them correctly, but a provenance field
  # that is right only because another one is checked is not provenance.
  export STABLE_ENVS_GIT_COMMIT=$(cat "$1/GIT_COMMIT" 2>/dev/null || true)
  PYTHONPATH=$1 "$SCR/venvs/$3/bin/python" $1/examples/bench_step3.py \
      --device cpu --threads 1 --envs "$4" --reps 15 --max-cell-s 12 \
      $5 "$ST" --out "$OUT/cpu_1thread_$2.json" || echo "SWEEP $2 rc=$?"
}

SWM=SimplePointMaze,TwoRoom,TwoRoomTarget,Piecewise,PiecewiseZones
run "$BEF" before_swm  se-step5 "$SWM"  --states-out
run "$AFT" after_swm   se-step5 "$SWM"  --states-in
run "$BEF" before_pusht se-step3 PushT  --states-out
run "$AFT" after_pusht  se-step3 PushT  --states-in

ls -la "$OUT"
echo "CPU_SWEEP_DONE"
