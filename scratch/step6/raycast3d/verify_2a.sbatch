#!/bin/bash
#SBATCH --job-name=se6-raycast3d-2a
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2a verification: the tendon FK is built for the whole batch instead of once per
# environment. BEFORE ($SCR/se6/baseline, never modified) and AFTER (this branch) in ONE job on ONE
# card, per AGENT_COMMON.
#
# Also carries the RAYCAST-2c follow-up probe: the qpos0 reset did not move the symptom, so the
# carrier is something else and bake_probe.py separates the candidates instead of guessing twice.
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
DUMP=$SCR/se6/_dumps/2a
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$DUMP"; mkdir -p "$DUMP/before" "$DUMP/after"

echo "######## 2c follow-up: what makes the bake non-reproducible?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/bake_probe.py --device cuda \
    --configs AntmazeGiantNavigate AntmazeTeleportExplore HumanoidmazeMediumNavigate \
    --out "$OUT/bake_probe_after_2c.json" || echo "PROBE rc=$?"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/bake_probe.py --device cuda \
    --configs AntmazeGiantNavigate AntmazeTeleportExplore \
    --out "$OUT/bake_probe_baseline.json" || echo "PROBE_BASE rc=$?"

echo "######## 2a: bit-identity of the batched tendon FK"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_dmcontrol_tendon.py -q -p no:randomly 2>&1 | tail -20 \
    || echo "UNIT rc=$?"

DMC="DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## 2a: replay the 10 DMControlSuite configs, before and after"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $DMC \
    --dump-dir "$DUMP/before" --out "$OUT/replay/2a_dmc_before_baseline.json" || echo "BEFORE rc=$?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $DMC \
    --dump-dir "$DUMP/after" --out "$OUT/replay/2a_dmc_after.json" || echo "AFTER rc=$?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$DUMP/before" "$DUMP/after" \
    --out "$OUT/replay/2a_dmc_pixel_diff.json" || echo "DIFF rc=$?"

echo "######## 2a: the mandatory GPU sweep, before and after, same card"
for E in DMCBallInCupCatch DMCFingerSpin; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2a/before_baseline" || echo "SWEEP_BEFORE $E rc=$?"
  PYTHONPATH=$WS $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --cprofile 256 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2a/after" || echo "SWEEP_AFTER $E rc=$?"
done

echo "2A done"
