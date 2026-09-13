#!/bin/bash
#SBATCH --job-name=se-s6-gpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# GPU profile-before for the two workstreams whose bottleneck is already well posed.
#
# fetch-gpu: R = 5.34 against a class-S bar of 4, so THROUGHPUT ALREADY PASSES and every one of the
# 12 slow rows is blocked by BATCH_SCALES_FAIL alone -- fps(4096) is only 7-12x fps(1) where 20x is
# required. That is a fixed-cost signature, so the sweep runs the full mandatory B list and reports
# set_state separately from render: per-env state unpack in Python is the classic culprit and
# folding it into render time is how it stays hidden.
#
# raycast3d/BallInCup: the one GPU row below tier on that path (GPU_PARITY, BELOW_G4,
# BATCH_SCALES_FAIL) while its siblings are GPU_FAST or GPU_EXCEPTIONAL -- an outlier on a shared
# path, which usually means something env-specific rather than something wrong with the caster.
#
# geforce3090 by name: Step 3's GPU baseline was measured on gpu2101/2107/2108, all 3090s. An L40S
# would satisfy a looser constraint and make the before/after a comparison between two machines.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$SCR/stable-envs
OUT=$SCR/step6
set -x
cd "$SCR/stable-envs"

# Fetch renders through the mujoco 3.11 stack Step 3B pinned; nothing else here can build it.
PY311=$SCR/venvs/se-s3b-fetch311/bin/python
for E in FetchPickAndPlace FetchReach; do
  $PY311 scratch/step6/profile_note.py --env $E --device cuda \
     --batch 1 16 64 256 1024 --repeat 10 --warmup 3 --workstream step6/fetch-gpu \
     --states-dir "$SCR/stable-envs/examples/results/step4/states" \
     --out "$OUT/fetch-gpu/profile_before" || echo "GPU $E rc=$?"
done

PY=$SCR/venvs/se-step6/bin/python
for E in DMCBallInCupCatch DMCReacherHard DMCFingerSpin; do
  $PY scratch/step6/profile_note.py --env $E --device cuda \
     --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --workstream step6/raycast3d \
     --states-dir "$SCR/stable-envs/examples/results/step4/states" \
     --out "$OUT/raycast3d/profile_before_gpu" || echo "GPU $E rc=$?"
done
echo "GPU_PROFILE done"
ls -la "$OUT"/*/profile_before*/ 2>/dev/null | tail -20
