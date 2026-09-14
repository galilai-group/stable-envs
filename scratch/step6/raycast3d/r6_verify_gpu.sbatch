#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6vgpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6 cuda verification. The fused cast is untouched and composite's tiled path is the same
# arithmetic on fewer samples, so the cuda digests cannot move -- which is a prediction, and this is
# the job that tests it. Both legs in ONE job on ONE card, defaults on both.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r6cm_gpu
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
rm -rf "$D"; mkdir -p "$D"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## cuda, branch vs integration, defaults both legs, all 21"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/int" --out "$OUT/replay/r6cm_cuda_integration.json" || echo "INT rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/br"  --out "$OUT/replay/r6cm_cuda_branch.json" || echo "BR rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/int" "$D/br" \
    --out "$OUT/replay/r6cm_cuda_identity.json" || echo "DIFF rc=$?"

echo "######## cuda, COMPOSITE_COMPACT_MERGE on == off"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --compact-merge off --dump-dir "$D/off" --out "$OUT/replay/r6cm_cuda_off.json" || echo "OFF rc=$?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/br" \
    --out "$OUT/replay/r6cm_cuda_switch_identity.json" || echo "SW rc=$?"

echo "######## the fast-path suite at the envs' own sizes"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly 2>&1 | tail -8

echo "######## GPU no-regression: the composite cells, both trees, same card"
for E in PointmazeLargeNavigate PointmazeMediumNavigate AntmazeMediumExplore; do
  for LEG in int br; do
    [ "$LEG" = int ] && T=$INT || T=$WS
    PYTHONPATH=$T $PY scratch/step6/raycast3d/profile_phases.py --env "$E" --device cuda \
        --batch 64 256 1024 --repeat 5 \
        --out "$OUT/experiments/r6cm_gpu_oneshot/${LEG}_${E}" 2>&1 | tail -8
    echo "GPU $LEG $E rc=$?"
  done
done
echo "R6_VGPU done"
