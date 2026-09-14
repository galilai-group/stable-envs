#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6rev
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Post-review verification on the HEAD that ships. The renderer is byte-identical to the one already
# replayed -- the review changed tests and prose only -- so this is a control, and a control that was
# not run would be an assumption.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r6rev
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv,noheader; cat GIT_COMMIT
rm -rf "$D"; mkdir -p "$D"
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/int" --out "$OUT/replay/r6rev_cuda_integration.json" || echo "INT rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL \
    --dump-dir "$D/br"  --out "$OUT/replay/r6rev_cuda_branch.json" || echo "BR rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/int" "$D/br" \
    --out "$OUT/replay/r6rev_cuda_identity.json" || echo "DIFF rc=$?"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
echo "R6_REV done"
