#!/bin/bash
#SBATCH --job-name=se6-texture-fidelity-vis311
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6A (texture-fidelity): does the MuJoCo 3.11 -> 3.12 renderer change explain Step 4's
# visual-tier failures on DMCCheetahRun / DMCHopperHop / DMCWalkerWalk (DMCHumanoidStand = control)?
#
# The declaration (stable_envs/envs/dmcontrol.py::_AT) was measured under mujoco 3.11.0 +
# mujoco-warp 3.11.0; Step 4 (job 6176998, gpu2105, RTX 3090) measured under mujoco 3.12.0 +
# mujoco_warp 3.12.0. The compiled texture bytes (tex_data) are byte-identical across the two
# versions (checked on the Mac), so any official-frame difference is libmujoco's GL renderer.
#
# BOTH arms run in THIS job, sequentially, on the same node and card:
#   arm 3.11 : $SCR/venvs/se-vis311  (built here if missing: mujoco 3.11.0, dm_control 1.0.44,
#              mujoco-warp 3.11.0, torch 2.14.0+cu126)
#   arm 3.12 : $SCR/venvs/se-step5   (mujoco 3.12.0, dm_control 1.0.45, mujoco_warp 3.12.0)
# Each arm: (1) examples/fidelity_step4.py replay of the frozen Step-4 corpora -> the Gate-A
# numbers under that version; (2) localize.py -> per-surface residual off MuJoCo's segmentation.
# Then cross.py compares the two arms pixel for pixel. The frozen corpora are READ-ONLY; every
# output goes under examples/results/step6/texture-fidelity/.
#
# This is the norm-gpu twin of scratch/step6/visual311.sbatch (job 6328158, stuck on the condo
# QOS). Two fixes over that script: --configs is comma-separated (argparse takes one string), and
# PYTHONPATH points at the workstream copy rather than the stale $SCR/stable-envs checkout.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/texture-fidelity
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
export UV_PYTHON_INSTALL_DIR=$SCR/.uv-python
export TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp HF_HOME=$SCR/.hf
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$WS
OUT=$WS/examples/results/step6/texture-fidelity
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
VENV311=$SCR/venvs/se-vis311
VENV312=$SCR/venvs/se-step5
mkdir -p "$OUT" "$SCR/logs"
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd "$WS"

if [ ! -x "$VENV311/bin/python" ]; then
  uv venv --python 3.12 "$VENV311" || exit 1
  uv pip install --python "$VENV311/bin/python" \
     --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
     --index-strategy unsafe-best-match \
     "torch==2.14.0+cu126" "mujoco==3.11.0" "dm_control==1.0.44" \
     "mujoco-warp==3.11.0" "warp-lang" "gymnasium>=1.0" numpy scipy imageio pillow || exit 1
fi

for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$VENV311/bin/python; else PY=$VENV312/bin/python; fi
  "$PY" -c "import mujoco, dm_control, mujoco_warp, warp, torch, scipy, PIL, importlib.metadata as md; print('ARM $ARM mujoco', mujoco.__version__, 'dm_control', md.version('dm_control'), 'mujoco_warp', md.version('mujoco-warp'), 'warp', warp.__version__, 'torch', torch.__version__)" || { echo "ARM $ARM import failed"; continue; }
  mkdir -p "$OUT/visual$ARM"
  "$PY" examples/fidelity_step4.py --configs "$CONFIGS" --device cuda \
      --states-dir "$STATES" --out "$OUT/visual$ARM/dmc.json" \
      --images "$OUT/visual$ARM/comparisons"
  echo "REPLAY $ARM rc=$?"
  "$PY" scratch/step6/texture-fidelity/localize.py --configs "$CONFIGS" \
      --states-dir "$STATES" --out "$OUT/localize$ARM"
  echo "LOCALIZE $ARM rc=$?"
  "$PY" scratch/step6/texture-fidelity/resprobe.py --configs "$CONFIGS" \
      --states-dir "$STATES" --out "$OUT/resprobe$ARM.json"
  echo "RESPROBE $ARM rc=$?"
  if [ "$ARM" = 312 ]; then
    # ablation: stock mujoco_warp lighting (no stable-envs lightspec specialisation) under 3.12
    STABLE_ENVS_LIGHTSPEC=0 "$PY" scratch/step6/texture-fidelity/localize.py --configs "$CONFIGS" \
        --states-dir "$STATES" --out "$OUT/localize312_nolightspec"
    echo "LOCALIZE 312 nolightspec rc=$?"
  fi
done

"$VENV312/bin/python" scratch/step6/texture-fidelity/cross.py \
    --a "$OUT/localize311" --b "$OUT/localize312" --out "$OUT/cross"
echo "CROSS rc=$?"
ls -la "$OUT" "$OUT"/visual311 "$OUT"/visual312 "$OUT"/localize311 "$OUT"/localize312 "$OUT"/cross 2>/dev/null
echo "VIS311 DONE"
