#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r4gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-4's GPU no-regression check. The bound is in the TORCH cast only and _launch's fused
# kernel is untouched, so the cuda digests must be unchanged -- this is the control that says so
# rather than the claim that they are.
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
D=$SCR/se6/_dumps/r4gpu
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$D"; mkdir -p "$D"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## cuda: all 21 configs, baseline vs branch, digests and pixels"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/before" --out "$OUT/replay/r4_cuda_before_baseline.json" || echo "B rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/after" --out "$OUT/replay/r4_cuda_after.json" || echo "A rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/before" "$D/after" \
    --out "$OUT/replay/r4_cuda_pixel_diff.json" || echo "D rc=$?"

echo "######## the bit-identity switches, on the envs"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly \
    -k "primitive-bounds or composite-cover-compaction or mover-window or raycast3d-fused or declared_fast_path" \
    2>&1 | tail -12 || echo "FASTPATH rc=$?"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_raycast3d_primitive_bounds.py -q -p no:randomly \
    2>&1 | tail -5 || echo "FUZZ rc=$?"

echo "######## GPU no-regression sweep (the fused path must be untouched)"
for E in DMCBallInCupCatch DMCFingerSpin; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 64 256 4096 --repeat 10 --warmup 3 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_r4/gpu_before_baseline" || echo "GB $E rc=$?"
  PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 64 256 4096 --repeat 10 --warmup 3 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_r4/gpu_after" || echo "GA $E rc=$?"
done
echo "R4_GPU done"
