#!/bin/bash
#SBATCH --job-name=se6-raycast3d-merged
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The final RAYCAST-2 verification, ON THE MERGED CODE (step6/integration merged into
# step6/raycast3d). Everything that gated a sub-ticket is re-run here so no result in the report
# describes a tree that no longer exists: all 21 live raycast3d configs replayed against the frozen
# Step-4 digests, both bit-identity switches diffed on the envs themselves, and the BallInCup sweep
# re-confirmed against the pristine baseline on the same card.
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
D=$SCR/se6/_dumps/merged
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$D"; mkdir -p "$D/before" "$D/after"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## all 21 live raycast3d configs, baseline vs merged branch, frozen digests"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/before" --out "$OUT/replay/merged_all21_before_baseline.json" || echo "B rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/after" --out "$OUT/replay/merged_all21_after.json" || echo "A rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/before" "$D/after" \
    --out "$OUT/replay/merged_all21_pixel_diff.json" || echo "D rc=$?"

echo "######## the two bit-identity switches, on the envs"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_dmcontrol_tendon.py -q -p no:randomly 2>&1 | tail -8 \
    || echo "UNIT rc=$?"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly \
    -k "composite-cover-compaction or mover-window or raycast3d-fused or declared_fast_path" \
    2>&1 | tail -10 || echo "FASTPATH rc=$?"

echo "######## the mandatory GPU sweep on the merged code, same card as its baseline"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env DMCBallInCupCatch --device cuda \
    --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
    --states-dir "$WS/examples/results/step4/states" \
    --out "$OUT/profile_after_2a/merged_before_baseline" || echo "SB rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env DMCBallInCupCatch --device cuda \
    --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --cprofile 256 \
    --states-dir "$WS/examples/results/step4/states" \
    --out "$OUT/profile_after_2a/merged_after" || echo "SA rc=$?"

echo "MERGED done"
