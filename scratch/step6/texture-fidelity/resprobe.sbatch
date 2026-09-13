#!/bin/bash
#SBATCH --job-name=se6-texture-fidelity-resprobe
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6A follow-up: check_resolution_consistency under the BOUND'S OWN protocol (8 states from
# render_worker.STATE_SEED) versus the frozen 64-state corpus, on both MuJoCo arms. The check never
# reads the reference renderer, so this separates "which states" from "which MuJoCo".
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/texture-fidelity
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$WS
OUT=$WS/examples/results/step6/texture-fidelity
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd "$WS"
for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$SCR/venvs/se-vis311/bin/python; else PY=$SCR/venvs/se-step5/bin/python; fi
  "$PY" scratch/step6/texture-fidelity/resprobe.py --configs "$CONFIGS" \
      --states-dir "$STATES" --out "$OUT/resprobe$ARM.json"
  echo "RESPROBE $ARM rc=$?"
done
echo "RESPROBE DONE"
