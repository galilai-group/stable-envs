#!/bin/bash
#SBATCH --job-name=se6-raycast3d-2c
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2c verification: OGBenchMaze._bake now resets qpos to qpos0.
#
# BEFORE and AFTER in ONE job on ONE card, per AGENT_COMMON: the "before" is the pristine baseline
# copy at $SCR/se6/baseline (never modified), the "after" is this branch. Both replay the SAME
# frozen Step-4 states and both are diffed against the frozen digests AND against each other in
# pixels, because "the digest moved" is not a finding until you can say which pixels and by how much.
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
DUMP=$SCR/se6/_dumps/2c
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$DUMP"; mkdir -p "$DUMP/before" "$DUMP/after"

OGB="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate"

# BEFORE: the pristine baseline. The replay script itself comes from this branch (it is a harness,
# not renderer code), but the stable_envs it imports is the baseline's.
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$DUMP/before" --out "$OUT/replay/2c_ogb_before_baseline.json" || echo "BEFORE rc=$?"

# AFTER: this branch.
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$DUMP/after" --out "$OUT/replay/2c_ogb_after.json" || echo "AFTER rc=$?"

PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$DUMP/before" "$DUMP/after" \
    --out "$OUT/replay/2c_ogb_pixel_diff.json" || echo "DIFF rc=$?"

# The Gate A check that failed, under the forced-cache-clear scenario that reproduced observed=1.
PYTHONPATH=$WS $PY scratch/step6/raycast3d/style_bisect.py --device cuda --batch 64 \
    --configs AntmazeTeleportExplore HumanoidmazeMediumNavigate \
              AntmazeTeleportNavigate HumanoidmazeLargeNavigate \
    --out "$OUT/style_is_state_bisect_after_2c.json" || echo "BISECT rc=$?"

# The new test, plus the check it sits beside, on the affected family.
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py -q -p no:randomly \
    -k "(plate_cache_eviction or style_is_state) and (Pointmaze or Antmaze or Humanoidmaze)" \
    2>&1 | tail -25 || echo "PYTEST rc=$?"

echo "2C done"
