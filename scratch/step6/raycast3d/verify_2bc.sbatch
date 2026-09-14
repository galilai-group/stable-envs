#!/bin/bash
#SBATCH --job-name=se6-raycast3d-2bc
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2b (composite cover-compaction) and RAYCAST-2c (bake pose), plus the RE-MEASURED 2a sweep.
#
# The previous run's before/after was invalid: the Step 6C scripts inserted their own repo root ahead
# of $PYTHONPATH, so BOTH legs imported the branch. They append it now and every note records the
# stable_envs it loaded, so "before" here really is $SCR/se6/baseline.
#
# 2b and 2c are verified together on the frozen corpus (one replay covers both) and separated by
# their own switches: COVER_COMPACTION isolates 2b exactly, and bake_probe.py isolates 2c.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D2A=$SCR/se6/_dumps/2a2; D2B=$SCR/se6/_dumps/2bc
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$D2A" "$D2B"; mkdir -p "$D2A/before" "$D2A/after" "$D2B/before" "$D2B/after"

OGB="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate"
DMC="DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## 2b+2c: replay the 11 OGBench configs, TRUE before vs after"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$D2B/before" --out "$OUT/replay/2bc_ogb_before_baseline.json" || echo "BEFORE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$D2B/after" --out "$OUT/replay/2bc_ogb_after.json" || echo "AFTER rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$D2B/before" "$D2B/after" --out "$OUT/replay/2bc_ogb_pixel_diff.json" \
    || echo "DIFF rc=$?"

echo "######## 2b+2c: the DM Control control set must not move either (composite is shared code)"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $DMC \
    --dump-dir "$D2A/before" --out "$OUT/replay/2bc_dmc_before_baseline.json" || echo "B2 rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $DMC \
    --dump-dir "$D2A/after" --out "$OUT/replay/2bc_dmc_after.json" || echo "A2 rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$D2A/before" "$D2A/after" --out "$OUT/replay/2bc_dmc_pixel_diff.json" \
    || echo "D2 rc=$?"

echo "######## 2b: the compaction is the identity, on the envs themselves"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly \
    -k "composite-cover-compaction or declared_fast_path" 2>&1 | tail -20 || echo "FASTPATH rc=$?"

echo "######## 2c: what actually makes the bake non-reproducible"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/bake_probe.py --device cuda \
    --configs AntmazeGiantNavigate AntmazeTeleportExplore HumanoidmazeMediumNavigate \
    --out "$OUT/bake_probe_branch.json" || echo "PROBE rc=$?"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/bake_probe.py --device cuda \
    --configs AntmazeGiantNavigate AntmazeTeleportExplore HumanoidmazeMediumNavigate \
    --out "$OUT/bake_probe_baseline.json" || echo "PROBE_BASE rc=$?"

echo "######## 2a: the mandatory GPU sweep, RE-MEASURED, true before vs after, same card"
for E in DMCBallInCupCatch DMCFingerSpin; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2a/before_baseline" || echo "SB $E rc=$?"
  PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --cprofile 256 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2a/after" || echo "SA $E rc=$?"
done

echo "2BC done"
