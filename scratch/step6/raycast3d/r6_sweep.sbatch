#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6sweep
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6's measurement, in two parts, on ONE node in ONE job at ONE thread.
#
# Part 1, INTERLEAVED off/on/off/on on the same branch. off2/off1 is a ratio of two runs of
# IDENTICAL code and therefore measures nothing but the harness; RAYCAST-5 measured that at up to
# 11.7% on a shared batch node, which is why a single before/after pair taken end-to-end is not
# evidence at this resolution. Four legs, and the signal is the mean of the two on-legs over the
# mean of the two off-legs.
#
# Part 2, the Step-3-shaped pair the dashboard can classify, with the official renderer PAIRED
# (a reference burst between every batch cell, median with its spread) and the decode bar taken on
# this node. examples/results/step6/gfxdraw/characterize/cpu_shard_attempts.json is why: a
# block-timed denominator on a shared node moved 2.3x inside one job and relabelled four rows.
# --max-reference-spread 1.25 rejects the node and writes the rejection rather than publishing.
#
# The BEFORE tree is $SCR/se6/integration -- the tree this branch was merged onto -- not the
# pre-Step-6 baseline. The ten envs are the eight RAYCAST-5's shard carried plus the two the ticket
# adds, so this pair supersedes that one rather than sitting beside it as a second after-leg.
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
OUT=$WS/examples/results/step6/raycast3d
SW=$OUT/sweep/r6
# rsync drops .git, so the SHA under measurement can only reach the shard through the environment.
# Without this the shard records tree_commit "unknown" and git_commit null -- which is what the first
# pair did, and why sweep/r6/README.md has to carry the commits in prose. The sync writes
# $WS/GIT_COMMIT from `git rev-parse HEAD`.
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/GIT_COMMIT" 2>/dev/null || echo unknown)
mkdir -p "$SW" "$OUT/profile_after_r6"
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
cat /proc/loadavg

echo "######## part 1: interleaved off/on/off/on, same branch, same node"
for E in PointmazeLargeNavigate PointmazeMediumNavigate AntmazeGiantNavigate HumanoidmazeMediumNavigate; do
  for LEG in off1 on1 off2 on2; do
    S=${LEG%[12]}
    PYTHONPATH=$WS $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
        --batch 1 16 64 256 --repeat 15 --warmup 4 --compact-merge $S \
        --states-dir "$WS/examples/results/step4/states" \
        --out "$OUT/profile_after_r6/interleaved_$LEG" || echo "SWEEP $E $LEG rc=$?"
  done
done

echo "######## part 2: the Step-3-shaped pair, paired denominator, same node"
ENVS="PointmazeLargeNavigate,PointmazeMediumNavigate,AntmazeGiantNavigate,HumanoidmazeMediumNavigate,DMCReacherHard,DMCFingerSpin,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCBallInCupCatch"
SE6_LEG=before PYTHONPATH=$INT $PY scratch/step6/raycast3d/r6_certify_shard.py \
    --device cpu --threads 1 --envs "$ENVS" --reps 25 --max-cell-s 20 \
    --max-reference-spread 1.25 \
    --out "$SW/cpu_1thread_before_integration.json" || echo "SHARD_BEFORE rc=$?"
SE6_LEG=after PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_certify_shard.py \
    --device cpu --threads 1 --envs "$ENVS" --reps 25 --max-cell-s 20 \
    --max-reference-spread 1.25 \
    --out "$SW/cpu_1thread_after_branch.json" || echo "SHARD_AFTER rc=$?"

# The tiers are read with scratch/step6/raycast3d/r6_tiers.py instead of by running
# scratch/step3/certify.py here. certify.py takes no --doc (that flag was copied from another
# workstream's script and made this step exit 2), and more importantly it REWRITES
# docs/research/step3_benchmarks.md as a side effect -- a Step-3 document this workstream must not
# touch. Run it out of tree if a label is wanted from the certifier itself.
echo "######## tiers, read off the shard pair"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_tiers.py 2>&1 | tail -20 || echo "TIERS rc=$?"
echo "R6_SWEEP done"
