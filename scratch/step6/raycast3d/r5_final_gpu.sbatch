#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5finalgpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r5finalgpu
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$D"; mkdir -p "$D"
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/before" --out "$OUT/replay/r5final_cuda_baseline.json" || echo "B rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/after" --out "$OUT/replay/r5final_cuda_branch.json" || echo "A rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/before" "$D/after" \
    --out "$OUT/replay/r5final_cuda_identity.json" || echo "D rc=$?"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly \
    -k "primitive-bounds or composite-cover-compaction or mover-window or raycast3d-fused or declared_fast_path" \
    2>&1 | tail -6 || echo "FASTPATH rc=$?"
echo "R5_FINAL_GPU done"
