#!/bin/bash
#SBATCH --job-name=se6-texture-fidelity-arms
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6A, second job. Job 6343431 built se-vis311 without `gymnasium` (a core dependency in
# pyproject.toml), so its 3.11 arm imported nothing; the 3.12 arm completed. gymnasium 1.3.0 has since
# been installed into se-vis311 (login node, uv pip). This job runs on ONE node/card:
#   3.11 arm : replay (fidelity_step4.py) + localize + resprobe
#   3.12 arm : localize (again, so the cross-arm comparison is same-card) + resprobe
#   cross    : the two arms pixel for pixel
# The 3.12 replay of job 6343431 (visual312/) stands as the certified-arm replay; the 3.12 localize
# here overwrites localize312/ so cross/ compares frames from the same card.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/texture-fidelity
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$WS
OUT=$WS/examples/results/step6/texture-fidelity
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
PY311=$SCR/venvs/se-vis311/bin/python
PY312=$SCR/venvs/se-step5/bin/python
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd "$WS"
for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$PY311; else PY=$PY312; fi
  "$PY" -c "import mujoco, dm_control, mujoco_warp, warp, torch, gymnasium, importlib.metadata as md; print('ARM $ARM mujoco', mujoco.__version__, 'dm_control', md.version('dm_control'), 'mujoco_warp', md.version('mujoco-warp'), 'warp', warp.__version__, 'torch', torch.__version__, 'gymnasium', gymnasium.__version__)" || { echo "ARM $ARM import failed"; continue; }
  if [ "$ARM" = 311 ]; then
    mkdir -p "$OUT/visual311"
    "$PY" examples/fidelity_step4.py --configs "$CONFIGS" --device cuda \
        --states-dir "$STATES" --out "$OUT/visual311/dmc.json" --images "$OUT/visual311/comparisons"
    echo "REPLAY 311 rc=$?"
  fi
  "$PY" scratch/step6/texture-fidelity/localize.py --configs "$CONFIGS" \
      --states-dir "$STATES" --out "$OUT/localize$ARM"
  echo "LOCALIZE $ARM rc=$?"
  "$PY" scratch/step6/texture-fidelity/resprobe.py --configs "$CONFIGS" \
      --states-dir "$STATES" --out "$OUT/resprobe$ARM.json"
  echo "RESPROBE $ARM rc=$?"
done
"$PY312" scratch/step6/texture-fidelity/cross.py --a "$OUT/localize311" --b "$OUT/localize312" --out "$OUT/cross"
echo "CROSS rc=$?"
ls -la "$OUT" "$OUT"/visual311 "$OUT"/localize311 "$OUT"/localize312 "$OUT"/cross 2>/dev/null | head -60
echo "ARMS DONE"
