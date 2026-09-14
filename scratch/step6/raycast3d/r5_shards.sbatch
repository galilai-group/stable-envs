#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5shardcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step-3-SHAPED shards for the Step-6 report generator, which labels tiers only from these and finds
# none under this workstream's directory.
#
# Produced by examples/bench_step3.py -- the Step-3 producer itself -- rather than back-converted
# from the profile_phases notes. The notes have fps and nothing else; a shard needs `decode` and
# `official_cpu` as well, and those are the DENOMINATORS of both tier bars. Synthesising them by
# copying the frozen node2349 row onto a measurement taken on a different machine would put a
# cross-host ratio into a file whose whole purpose is to be classified as if it were Step 3, and
# nothing downstream would be able to tell. Running the real producer measures se_fps, the official
# renderer and the JPEG decode rate on ONE host, which is what the bars are defined against.
#
# Both legs in one job on one node, baseline first, so the pair is comparable to each other even
# though neither is comparable to the frozen node2349 rows.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$SCR/tmp; mkdir -p "$TMPDIR"
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d/sweep
mkdir -p "$OUT"
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
cat /proc/loadavg

ENVS="DMCReacherHard,DMCFingerSpin,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCBallInCupCatch,PointmazeLargeNavigate,PointmazeMediumNavigate"

# THREE legs, because the release decision on PRIMITIVE_BOUNDS is the coordinator's and not mine.
# The branch ships with the bound OFF today; if it is turned on, the shard set must already contain
# the leg that describes that build, or the dashboard would label a tier from a configuration nobody
# is running.
PYTHONPATH=$BASE $PY examples/bench_step3.py --device cpu --threads 1 --envs "$ENVS" \
    --reps 25 --max-cell-s 12 --out "$OUT/cpu_1thread_before_baseline.json" || echo "BEFORE rc=$?"
SE6_PRIMITIVE_BOUNDS=off PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_bench_step3_bounds.py \
    --device cpu --threads 1 --envs "$ENVS" \
    --reps 25 --max-cell-s 12 --out "$OUT/cpu_1thread_branch_bounds_off_control.json" || echo "OFF rc=$?"
SE6_PRIMITIVE_BOUNDS=on  PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_bench_step3_bounds.py \
    --device cpu --threads 1 --envs "$ENVS" \
    --reps 25 --max-cell-s 12 --out "$OUT/cpu_1thread_after_branch_bounds_on.json" || echo "ON rc=$?"
echo "R5_SHARD_CPU done"
