#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6fin
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The identity, re-taken on the HEAD that ships. Between job 6366546 and this one the renderer
# changed by one blank line and the test tree gained the batch-invariance cases -- neither can move
# a pixel, which is exactly the sort of belief worth spending four minutes to stop believing.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r6fin
set -x
cd "$WS"
rm -rf "$D"; mkdir -p "$D"
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/int" --out "$OUT/replay/r6fin_cpu_integration.json" || echo "INT rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/br"  --out "$OUT/replay/r6fin_cpu_branch.json" || echo "BR rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/int" "$D/br" \
    --out "$OUT/replay/r6fin_cpu_branch_vs_integration.json" || echo "DIFF rc=$?"
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
echo "R6_FIN done"
